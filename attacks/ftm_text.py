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


def hotflip_ftm_attack(
    surrogate,
    text: str,
    true_label: int,
    target_label: int,
    exp_settings: dict,
    device: str = "cpu",
) -> Dict:
    """Strategy A: HotFlip-style discrete token substitution with FTM hooks.

    This attack directly mutates token IDs using gradient-guided substitutions,
    avoiding repeated embedding-to-token projections during optimization.
    """
    device = torch.device(device)
    max_change_ratio = exp_settings["max_word_change_ratio"]
    num_iter = int(exp_settings.get("hotflip_num_iterations", exp_settings["num_iterations"]))
    top_k_words = int(exp_settings.get("hotflip_top_k_words", 100))
    words_per_iter = int(exp_settings.get("hotflip_words_per_iter", 3))
    margin = float(exp_settings.get("target_margin", 0.0))

    inputs = surrogate.tokenize(text)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    original_ids = input_ids.clone()
    tokens = input_ids.clone()

    special_ids = get_special_token_ids(surrogate.tokenizer)
    embedding_layer = surrogate.get_embedding_layer()
    vocab_embeddings = surrogate.get_vocab_embeddings().detach()

    # Compute change budget on non-special tokens.
    _, total_non_special, _ = compute_word_changes(original_ids, original_ids, special_ids)
    max_changes = max(1, int(total_non_special * max_change_ratio)) if total_non_special > 0 else 0

    ftm_model = TextFeatureTuning(surrogate, exp_settings, device=str(device))

    # Record clean features once from original text.
    with torch.no_grad():
        clean_embeddings = embedding_layer(original_ids).detach()
        ftm_model.start_feature_record()
        ftm_model.forward_from_embeddings(clean_embeddings, attention_mask)
        ftm_model.end_feature_record()

    best_adv_ids = tokens.clone()
    best_adv_text = embeddings_to_text(tokens, surrogate.tokenizer)
    best_score = float("-inf")
    attack_success = False

    for t in range(1, num_iter + 1):
        current_embeddings = embedding_layer(tokens).detach().clone().requires_grad_(True)
        outputs = ftm_model.forward_from_embeddings(current_embeddings, attention_mask)
        logits = outputs.logits[0]

        target_logit = logits[target_label]
        other_logits = logits.clone()
        other_logits[target_label] = -1e9
        max_other_logit = torch.max(other_logits)
        loss = target_logit - max_other_logit - margin

        grad_embeddings = torch.autograd.grad(
            loss,
            current_embeddings,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grad_embeddings is None:
            continue

        grad_norm = grad_embeddings.norm(dim=-1)[0]
        seq_len = tokens.size(1)

        # Rank mutable positions by gradient norm.
        mutable_positions = []
        for pos in range(seq_len):
            token_id = tokens[0, pos].item()
            if token_id in special_ids:
                continue
            if attention_mask[0, pos].item() == 0:
                continue
            mutable_positions.append((pos, grad_norm[pos].item()))

        mutable_positions.sort(key=lambda x: x[1], reverse=True)
        selected_positions = [p for p, _ in mutable_positions[:max(1, words_per_iter)]]

        with torch.no_grad():
            for pos in selected_positions:
                current_id = tokens[0, pos].item()
                grad_vec = grad_embeddings[0, pos]

                # First-order approximation score for each candidate token.
                candidate_scores = torch.mv(vocab_embeddings, grad_vec)

                for sid in special_ids:
                    candidate_scores[sid] = -1e9
                candidate_scores[current_id] = -1e9

                k = min(max(2, top_k_words), candidate_scores.numel())
                top_ids = torch.topk(candidate_scores, k=k).indices

                accepted_id = None
                for cand_id in top_ids.tolist():
                    tmp_tokens = tokens.clone()
                    tmp_tokens[0, pos] = cand_id
                    changed, _, _ = compute_word_changes(original_ids, tmp_tokens, special_ids)
                    if changed <= max_changes:
                        accepted_id = cand_id
                        break

                if accepted_id is not None:
                    tokens[0, pos] = accepted_id

        with torch.no_grad():
            eval_logits = surrogate.model(input_ids=tokens, attention_mask=attention_mask).logits[0]
            pred = torch.argmax(eval_logits).item()
            eval_target = eval_logits[target_label].item()
            eval_other = eval_logits.clone()
            eval_other[target_label] = -1e9
            eval_margin = eval_target - torch.max(eval_other).item()
            _, _, change_ratio = compute_word_changes(original_ids, tokens, special_ids)

            if change_ratio <= max_change_ratio and eval_margin > best_score:
                best_score = eval_margin
                best_adv_ids = tokens.clone()
                best_adv_text = embeddings_to_text(best_adv_ids, surrogate.tokenizer)

            if pred == target_label and change_ratio <= max_change_ratio:
                attack_success = True
                best_adv_ids = tokens.clone()
                best_adv_text = embeddings_to_text(best_adv_ids, surrogate.tokenizer)
                break

            if t % 25 == 0:
                print(
                    f"  iter {t:>4d}/{num_iter} | "
                    f"pred={pred} target={target_label} | "
                    f"changed={change_ratio:.1%} | "
                    f"margin={eval_margin:.3f}"
                )

    ftm_model.remove_hooks()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with torch.no_grad():
        final_inputs = surrogate.tokenize(best_adv_text)
        final_pred = torch.argmax(surrogate.model(**final_inputs).logits, dim=-1).item()

    num_changed, _, change_ratio = compute_word_changes(original_ids, best_adv_ids, special_ids)

    return {
        "original_text": text,
        "adversarial_text": best_adv_text,
        "original_ids": original_ids,
        "adversarial_ids": best_adv_ids,
        "true_label": true_label,
        "target_label": target_label,
        "surrogate_pred": final_pred,
        "attack_success": attack_success,
        "num_changed": num_changed,
        "change_ratio": change_ratio,
    }


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
    use_momentum = bool(exp_settings.get("use_momentum", False))
    momentum_mu = float(exp_settings.get("momentum_mu", 0.9))
    adaptive_step = bool(exp_settings.get("adaptive_step", False))
    adaptive_check_freq = int(exp_settings.get("adaptive_check_freq", 50))
    adaptive_min_margin_gain = float(exp_settings.get("adaptive_min_margin_gain", 0.02))
    adaptive_step_scale = float(exp_settings.get("adaptive_step_scale", 1.5))
    alpha_max = float(exp_settings.get("alpha_max", 0.5))
    project_intermediate = bool(exp_settings.get("project_intermediate", True))

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
    original_embeddings = embeddings.detach().clone()

    # Non-special token mask used by change-push regularization.
    token_mask = torch.ones_like(input_ids, dtype=torch.float32, device=device)
    for sid in special_ids:
        token_mask = token_mask * (input_ids != sid).float()

    # ── 2. Wrap model with FTM hooks ─────────────────────────────────
    ftm_model = TextFeatureTuning(surrogate, exp_settings, device=str(device))

    # Record clean features (1 forward pass)
    with torch.no_grad():
        ftm_model.start_feature_record()
        ftm_model.forward_from_embeddings(embeddings, attention_mask)
        ftm_model.end_feature_record()

    # ── 3. Iterative attack ──────────────────────────────────────────
    embeddings = embeddings.detach().clone().requires_grad_(True)
    momentum = torch.zeros_like(embeddings)
    step_alpha = alpha
    last_margin_check = None
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
        current_margin_score = (target_logit - max_other_logit).item()

        # Encourage movement away from original token embeddings to avoid
        # near-identity projections when constraints are too conservative.
        change_push_weight = float(exp_settings.get("change_push_weight", 0.0))
        if change_push_weight > 0.0:
            emb_delta = (embeddings - original_embeddings).pow(2).mean(dim=-1)
            change_push = (emb_delta * token_mask).sum() / (token_mask.sum() + 1e-6)
            loss = loss + change_push_weight * change_push

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
                if use_momentum:
                    normalized_grad = grad_emb / (grad_emb.abs().mean() + 1e-8)
                    momentum = momentum_mu * momentum + normalized_grad
                    update_direction = momentum.sign()
                else:
                    update_direction = grad_emb.sign()

                embeddings = embeddings + step_alpha * update_direction
            embeddings = embeddings.detach().requires_grad_(True)

        # Optional adaptive step-size schedule when margin is stuck.
        if adaptive_step and (t % adaptive_check_freq == 0):
            if last_margin_check is not None:
                margin_gain = current_margin_score - last_margin_check
                if margin_gain < adaptive_min_margin_gain:
                    step_alpha = min(step_alpha * adaptive_step_scale, alpha_max)
            last_margin_check = current_margin_score

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
        should_project = (
            t == num_iter - 1
            or (
                project_intermediate
                and projection_freq is not None
                and projection_freq > 0
                and (t % projection_freq == 0)
            )
        )
        if should_project:
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
