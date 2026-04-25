"""
Main FTM attack loop for text classification.

Mirrors the image ``ftm_attack()`` function in ``attacks.py``:
  1. Tokenize input → get embeddings  (continuous surrogate for pixels)
  2. Record clean features via hooks
  3. 300-iter optimisation loop:
     • Forward with inputs_embeds  (hooks add Δz automatically)
     • Loss = logit of target class (targeted attack)
     • Gradients for embeddings + Δz
     • Sign-gradient update on embeddings
     • Gradient update on Δz for stochastically selected layers
     • Periodic projection to nearest vocab words
  4. Return adversarial text + metadata
"""

import torch
from typing import Dict

from attacks.hooks import TextFeatureTuning
from attacks.word_projection import (
    project_to_nearest_words,
    embeddings_to_text,
    get_special_token_ids,
    compute_word_changes,
)


def ftm_text_attack(
    surrogate,                 # SurrogateModel instance
    text: str,
    true_label: int,
    target_label: int,
    exp_settings: dict,
    device: str = "cpu",
) -> Dict:
    """Run the text FTM attack on a single sample.

    Args:
        surrogate:    loaded ``SurrogateModel``.
        text:         original input text.
        true_label:   ground-truth class index.
        target_label: class index we want the model to predict.
        exp_settings: configuration dictionary.
        device:       torch device string.

    Returns:
        dict with keys:
          ``original_text``, ``adversarial_text``,
          ``original_ids``, ``adversarial_ids``,
          ``true_label``, ``target_label``,
          ``surrogate_pred``, ``num_changed``, ``change_ratio``
    """
    device = torch.device(device)
    num_iter = exp_settings["num_iterations"]
    alpha = exp_settings["alpha"]
    projection_freq = exp_settings["projection_freq"]
    max_change_ratio = exp_settings["max_word_change_ratio"]

    # ── 1. Tokenize ──────────────────────────────────────────────────
    inputs = surrogate.tokenize(text)
    input_ids = inputs["input_ids"]               # [1, seq_len]
    attention_mask = inputs["attention_mask"]       # [1, seq_len]
    original_ids = input_ids.clone()

    special_ids = get_special_token_ids(surrogate.tokenizer)
    vocab_embeddings = surrogate.get_vocab_embeddings().detach()

    # Get initial embeddings
    embedding_layer = surrogate.get_embedding_layer()
    embeddings = embedding_layer(input_ids).detach().clone()  # [1, seq, hidden]

    # ── 2. Wrap model with FTM hooks ─────────────────────────────────
    ftm_model = TextFeatureTuning(surrogate, exp_settings, device=str(device))

    # Record clean features (1 forward pass)
    with torch.no_grad():
        ftm_model.start_feature_record()
        ftm_model.forward_from_embeddings(embeddings, attention_mask)
        ftm_model.end_feature_record()

    # ── 3. Iterative attack ──────────────────────────────────────────
    embeddings = embeddings.detach().clone().requires_grad_(True)
    best_adv_ids = original_ids.clone()
    best_adv_text = text
    attack_success = False
    best_score = float("-inf")

    for t in range(1, num_iter):  # start from 1 (0 was the recording pass)
        # Forward pass (hooks add Δz / mixing)
        outputs = ftm_model.forward_from_embeddings(embeddings, attention_mask)
        logits = outputs.logits  # [1, num_classes]

        # Loss: maximise target margin over strongest non-target class.
        # This is usually stronger than maximising target logit alone.
        target_logit = logits[0, target_label]
        other_logits = logits[0].clone()
        other_logits[target_label] = -1e9
        max_other_logit = torch.max(other_logits)
        margin = float(exp_settings.get("target_margin", 0.0))
        loss = target_logit - max_other_logit - margin

        # ── Collect all parameters for autograd ──────────────────
        params = [embeddings]
        active_layer_indices = []

        for layer_idx, was_triggered in ftm_model.mixing_triggered.items():
            params.append(ftm_model.perturbations[layer_idx])
            if was_triggered:
                active_layer_indices.append(layer_idx)

        grads = torch.autograd.grad(
            loss, params, retain_graph=False, create_graph=False,
            allow_unused=True,
        )

        grad_emb = grads[0]  # gradient for embeddings

        # ── Update embeddings (sign-gradient ascent) ─────────────
        with torch.no_grad():
            if grad_emb is not None:
                embeddings = embeddings + alpha * grad_emb.sign()
            embeddings = embeddings.detach().requires_grad_(True)

        # ── Update Δz for each hooked layer ──────────────────────
        grad_idx = 1
        for layer_idx in ftm_model.mixing_triggered.keys():
            grad_dz = grads[grad_idx]
            if grad_dz is not None and layer_idx in active_layer_indices:
                ftm_model.perturbations[layer_idx] = (
                    (ftm_model.perturbations[layer_idx] - grad_dz)
                    .detach()
                    .requires_grad_(True)
                )
            else:
                ftm_model.perturbations[layer_idx] = (
                    ftm_model.perturbations[layer_idx].detach().requires_grad_(True)
                )
            grad_idx += 1

        # ── Periodic word projection ─────────────────────────────
        if t % projection_freq == 0 or t == num_iter - 1:
            with torch.no_grad():
                proj_emb, proj_ids = project_to_nearest_words(
                    embeddings,
                    vocab_embeddings,
                    special_ids,
                    original_ids,
                    top_k=exp_settings.get("top_k_projection", 50),
                    original_swap_gap=float(exp_settings.get("projection_swap_gap", 0.01)),
                )
                embeddings = proj_emb.detach().requires_grad_(True)

                # Evaluate current adversarial text
                adv_text = embeddings_to_text(proj_ids, surrogate.tokenizer)
                _, _, change_ratio = compute_word_changes(
                    original_ids, proj_ids, special_ids,
                )

                # Check if attack succeeded
                adv_inputs = surrogate.tokenize(adv_text)
                adv_logits = surrogate.model(**adv_inputs).logits
                pred = torch.argmax(adv_logits, dim=-1).item()
                adv_target = adv_logits[0, target_label].item()
                adv_other = adv_logits[0].clone()
                adv_other[target_label] = -1e9
                adv_margin = adv_target - torch.max(adv_other).item()

                if change_ratio <= max_change_ratio and adv_margin > best_score:
                    best_score = adv_margin
                    best_adv_text = adv_text
                    best_adv_ids = proj_ids.clone()

                if pred == target_label and change_ratio <= max_change_ratio:
                    attack_success = True

                if t % (projection_freq * 2) == 0:
                    print(
                        f"  iter {t:>4d}/{num_iter} | "
                        f"pred={pred} target={target_label} | "
                        f"changed={change_ratio:.1%} | "
                        f"logit[target]={logits[0, target_label].item():.3f}"
                    )

    # ── 4. Final projection & return ─────────────────────────────────
    with torch.no_grad():
        final_emb, final_ids = project_to_nearest_words(
            embeddings,
            vocab_embeddings,
            special_ids,
            original_ids,
            top_k=exp_settings.get("top_k_projection", 50),
            original_swap_gap=float(exp_settings.get("projection_swap_gap", 0.01)),
        )
        final_text = embeddings_to_text(final_ids, surrogate.tokenizer)
        num_changed, num_total, change_ratio = compute_word_changes(
            original_ids, final_ids, special_ids,
        )

        # Check final prediction
        final_inputs = surrogate.tokenize(final_text)
        final_logits = surrogate.model(**final_inputs).logits
        final_pred = torch.argmax(final_logits, dim=-1).item()
        final_target = final_logits[0, target_label].item()
        final_other = final_logits[0].clone()
        final_other[target_label] = -1e9
        final_margin = final_target - torch.max(final_other).item()

        if change_ratio <= max_change_ratio and final_margin > best_score:
            best_score = final_margin
            best_adv_text = final_text
            best_adv_ids = final_ids.clone()

        # If the final text is better, use it
        if final_pred == target_label and change_ratio <= max_change_ratio:
            attack_success = True

    ftm_model.remove_hooks()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    num_changed, _, change_ratio = compute_word_changes(
        original_ids, best_adv_ids, special_ids,
    )

    with torch.no_grad():
        best_inputs = surrogate.tokenize(best_adv_text)
        best_logits = surrogate.model(**best_inputs).logits
        surrogate_pred = torch.argmax(best_logits, dim=-1).item()

    return {
        "original_text": text,
        "adversarial_text": best_adv_text,
        "original_ids": original_ids,
        "adversarial_ids": best_adv_ids,
        "true_label": true_label,
        "target_label": target_label,
        "surrogate_pred": surrogate_pred,
        "attack_success": attack_success,
        "num_changed": num_changed,
        "change_ratio": change_ratio,
    }
