"""
Evaluation utilities: Attack Success Rate (ASR) and semantic similarity.
"""

import torch
import numpy as np
from typing import List, Dict, Optional


# ── Attack Success Rate ──────────────────────────────────────────────

class EvalResult:
    """Track per-model attack success rate (mirrors image FTM's EvalResult)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_samples = 0
        self.success_samples = 0

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.success_samples / self.total_samples * 100

    def update(self, pred_label: int, target_label: int):
        self.total_samples += 1
        if pred_label == target_label:
            self.success_samples += 1

    def __str__(self):
        return (
            f"Model: {self.model_name:<45} | "
            f"ASR: {self.success_rate:5.1f}% "
            f"({self.success_samples}/{self.total_samples})"
        )


def compute_asr(
    adversarial_texts: List[str],
    target_labels: List[int],
    black_box_models: list,
) -> Dict[str, EvalResult]:
    """Compute attack success rate on each black-box model.

    Args:
        adversarial_texts: list of adversarial text strings.
        target_labels:     corresponding target class indices.
        black_box_models:  list of ``BlackBoxModel`` instances.

    Returns:
        ``{model_name: EvalResult}``
    """
    results = {m.model_name: EvalResult(m.model_name) for m in black_box_models}

    for adv_text, target in zip(adversarial_texts, target_labels):
        for model in black_box_models:
            pred = model.predict(adv_text)
            results[model.model_name].update(pred, target)

    return results


# ── Semantic Similarity ──────────────────────────────────────────────

_sim_model = None  # lazy singleton


def _get_sim_model():
    global _sim_model
    if _sim_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence similarity model (all-MiniLM-L6-v2) ...")
        _sim_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sim_model


def compute_semantic_similarity(
    original_text: str,
    adversarial_text: str,
) -> float:
    """Cosine similarity between sentence embeddings (0-1 scale)."""
    sim_model = _get_sim_model()
    embs = sim_model.encode([original_text, adversarial_text])
    cos_sim = np.dot(embs[0], embs[1]) / (
        np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
    )
    return float(cos_sim)


def compute_batch_similarity(
    originals: List[str],
    adversarials: List[str],
) -> List[float]:
    """Compute pairwise cosine similarity for a batch."""
    sim_model = _get_sim_model()
    emb_orig = sim_model.encode(originals)
    emb_adv = sim_model.encode(adversarials)

    sims = []
    for o, a in zip(emb_orig, emb_adv):
        cos = np.dot(o, a) / (np.linalg.norm(o) * np.linalg.norm(a) + 1e-8)
        sims.append(float(cos))
    return sims
