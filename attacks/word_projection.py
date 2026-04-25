"""
Embedding → text projection utilities.

Since text tokens are discrete, we optimise in continuous embedding space and
periodically snap each position back to its nearest vocabulary token (by
cosine similarity).
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def project_to_nearest_words(
    embeddings: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    special_token_ids: set,
    original_ids: torch.Tensor,
    top_k: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project each embedding position to the nearest vocabulary word.

    Args:
        embeddings:       ``[1, seq_len, hidden]`` — current optimised embeddings.
        vocab_embeddings: ``[vocab_size, hidden]`` — full vocabulary matrix.
        special_token_ids: set of token IDs to skip (CLS, SEP, PAD, UNK).
        original_ids:     ``[1, seq_len]`` — original token IDs (special tokens
                          are preserved unchanged).
        top_k:            only consider the *top_k* nearest neighbours
                          (not used to restrict, kept for potential constraints).

    Returns:
        ``(new_embeddings [1, seq_len, hidden],
          new_ids        [1, seq_len])``
    """
    seq_len = embeddings.size(1)
    device = embeddings.device

    new_ids = original_ids.clone()
    new_emb_list = []

    # Normalise vocab for fast cosine sim  [vocab, hidden]
    vocab_norm = F.normalize(vocab_embeddings, dim=1)

    for i in range(seq_len):
        token_id = original_ids[0, i].item()

        if token_id in special_token_ids:
            # Keep special tokens unchanged
            new_emb_list.append(vocab_embeddings[token_id].unsqueeze(0))
            continue

        # Cosine similarity between this position and all vocab embeddings
        emb_norm = F.normalize(embeddings[0, i].unsqueeze(0), dim=1)  # [1, hidden]
        sims = torch.mm(emb_norm, vocab_norm.t()).squeeze(0)          # [vocab]

        # Mask out special tokens
        for sid in special_token_ids:
            sims[sid] = -1.0

        # Find nearest word
        best_id = sims.argmax().item()
        new_ids[0, i] = best_id
        new_emb_list.append(vocab_embeddings[best_id].unsqueeze(0))

    new_embeddings = torch.stack(new_emb_list, dim=1)  # [1, seq_len, hidden]
    return new_embeddings.to(device), new_ids.to(device)


def embeddings_to_text(
    token_ids: torch.Tensor,
    tokenizer,
) -> str:
    """Decode a tensor of token IDs back to a human-readable string."""
    return tokenizer.decode(token_ids.squeeze().tolist(), skip_special_tokens=True)


def get_special_token_ids(tokenizer) -> set:
    """Return the set of special-token IDs that should never be modified."""
    special_ids = set()
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id",
                 "unk_token_id", "mask_token_id", "bos_token_id",
                 "eos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    return special_ids


def compute_word_changes(
    original_ids: torch.Tensor,
    adversarial_ids: torch.Tensor,
    special_token_ids: set,
) -> Tuple[int, int, float]:
    """Count how many non-special tokens were changed.

    Returns:
        ``(num_changed, num_total_non_special, change_ratio)``
    """
    orig = original_ids.squeeze().tolist()
    adv = adversarial_ids.squeeze().tolist()

    total = 0
    changed = 0
    for o, a in zip(orig, adv):
        if o in special_token_ids:
            continue
        total += 1
        if o != a:
            changed += 1

    ratio = changed / total if total > 0 else 0.0
    return changed, total, ratio
