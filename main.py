"""
Text FTM Attack — CLI entry point.

Usage:
    python main.py --num_samples 5 --device cpu
    python main.py --num_samples 100 --device cuda:0 --eval
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime

import torch
import numpy as np

from config import exp_configuration
from models.surrogate_model import SurrogateModel
from models.black_box_models import load_black_box_models
from attacks.ftm_text import ftm_text_attack
from utils.data_loader import load_imdb_dataset, load_custom_csv
from utils.evaluation import (
    compute_asr,
    compute_semantic_similarity,
    compute_batch_similarity,
    EvalResult,
)


def main(args):
    # ── Configuration ────────────────────────────────────────────────
    settings = exp_configuration[args.config_idx]
    device = args.device

    # Override from CLI
    settings["num_samples"] = args.num_samples
    settings["seed"] = args.seed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  TEXT FTM ATTACK")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Surrogate    : {settings['surrogate_model']}")
    print(f"Iterations   : {settings['num_iterations']}")
    print(f"Samples      : {args.num_samples}")
    print(f"Eval on BBMs : {args.eval}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────
    if args.data_csv:
        samples = load_custom_csv(
            args.data_csv,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        samples = load_imdb_dataset(
            num_samples=args.num_samples,
            seed=args.seed,
        )

    # ── Load surrogate model ─────────────────────────────────────────
    print(f"\nLoading surrogate model: {settings['surrogate_model']}")
    surrogate = SurrogateModel(settings["surrogate_model"], device=device)
    print(f"  Transformer layers: {surrogate.num_layers}")
    print(f"  Hidden size:        {surrogate.hidden_size}")
    print(f"  Hooked layers:      {settings['target_layers']}")

    # ── Run attack ───────────────────────────────────────────────────
    results = []
    originals = []
    adversarials = []
    target_labels_list = []
    surrogate_successes = 0

    for i, (text, true_label, target_label) in enumerate(samples):
        print(f"\n{'─' * 60}")
        print(f"Sample {i + 1}/{len(samples)}")
        print(f"  True label  : {true_label} ({surrogate.LABEL_MAP.get(true_label, '?')})")
        print(f"  Target label: {target_label} ({surrogate.LABEL_MAP.get(target_label, '?')})")
        print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        result = ftm_text_attack(
            surrogate=surrogate,
            text=text,
            true_label=true_label,
            target_label=target_label,
            exp_settings=settings,
            device=device,
        )

        results.append(result)
        originals.append(text)
        adversarials.append(result["adversarial_text"])
        target_labels_list.append(target_label)

        if result["attack_success"]:
            surrogate_successes += 1

        print(f"\n  Adversarial : {result['adversarial_text'][:100]}{'...' if len(result['adversarial_text']) > 100 else ''}")
        print(f"  Surr. pred  : {result['surrogate_pred']}")
        print(f"  Words changed: {result['num_changed']} ({result['change_ratio']:.1%})")
        print(f"  Success     : {'✓' if result['attack_success'] else '✗'}")

    # ── Semantic similarity ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SEMANTIC SIMILARITY")
    print("=" * 60)
    similarities = compute_batch_similarity(originals, adversarials)
    for i, sim in enumerate(similarities):
        result = results[i]
        print(f"  [{i+1:>3d}] sim={sim:.3f} | changed={result['change_ratio']:.1%} | {'✓' if result['attack_success'] else '✗'}")

    avg_sim = np.mean(similarities)
    print(f"\n  Average similarity: {avg_sim:.3f}")
    above_threshold = sum(1 for s in similarities if s >= settings["semantic_sim_threshold"])
    print(f"  Above threshold ({settings['semantic_sim_threshold']}): {above_threshold}/{len(similarities)}")

    # ── Surrogate ASR ────────────────────────────────────────────────
    surr_asr = surrogate_successes / len(samples) * 100
    print(f"\n  Surrogate ASR: {surr_asr:.1f}% ({surrogate_successes}/{len(samples)})")

    # ── Black-box evaluation ─────────────────────────────────────────
    if args.eval:
        print(f"\n{'=' * 60}")
        print("BLACK-BOX TRANSFERABILITY")
        print("=" * 60)
        print("Loading black-box models...")
        bbm_list = load_black_box_models(settings["target_model_names"], device=device)

        asr_results = compute_asr(adversarials, target_labels_list, bbm_list)
        for name, er in asr_results.items():
            print(f"  {er}")

        avg_asr = np.mean([er.success_rate for er in asr_results.values()])
        print(f"\n  Average black-box ASR: {avg_asr:.1f}%")

    # ── Save results ─────────────────────────────────────────────────
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Save examples
    examples_path = os.path.join(save_dir, "examples.txt")
    with open(examples_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(f"Original ({r['true_label']}): {r['original_text']}\n")
            f.write(f"Adversarial ({r['surrogate_pred']}): {r['adversarial_text']}\n")
            f.write(f"Target: {r['target_label']} | Success: {r['attack_success']}\n")
            f.write(f"Words changed: {r['num_changed']} ({r['change_ratio']:.1%})\n")
            f.write(f"Similarity: {similarities[i]:.3f}\n\n")

    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in settings.items() if not callable(v)},
        "num_samples": len(samples),
        "surrogate_asr": surr_asr,
        "avg_semantic_similarity": float(avg_sim),
        "above_sim_threshold": above_threshold,
    }
    if args.eval:
        summary["black_box_asr"] = {
            name: er.success_rate for name, er in asr_results.items()
        }
        summary["avg_black_box_asr"] = float(avg_asr)

    summary_path = os.path.join(save_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {save_dir}/")
    print(f"  examples.txt        — adversarial text examples")
    print(f"  results_summary.json — metrics summary")
    print("=" * 60)
    print("DONE")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text FTM Attack — Feature Tuning Mixup for Text Classification"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of samples to attack",
    )
    parser.add_argument(
        "--config_idx", type=int, default=1,
        help="Config index from exp_configuration",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate transferability on black-box models",
    )
    parser.add_argument(
        "--data_csv", type=str, default="./data/IMDB Dataset.csv",
        help="Path to a custom CSV file (columns: review, sentiment). "
             "If not provided, uses HuggingFace IMDB dataset.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results",
        help="Directory to save results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
