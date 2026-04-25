"""
Perturbation (Δz) management utilities.

• Initialisation  — zeros matching hidden-state shapes
• Scaling         — norm-aware scaling (Eq. 11 from FTM paper)
• Gradient update — update selected layers' Δz using autograd gradients
"""

import torch
from typing import Dict, List


def init_perturbations(
    clean_features: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Create zero-initialised perturbation tensors for each recorded layer.

    Args:
        clean_features: ``{layer_idx: hidden_state_tensor}`` from recording pass.

    Returns:
        ``{layer_idx: zeros_like(hidden_state).requires_grad_(True)}``
    """
    perturbations = {}
    for idx, feat in clean_features.items():
        perturbations[idx] = torch.zeros_like(feat).detach().requires_grad_(True)
    return perturbations


def scale_perturbation(
    hidden: torch.Tensor,
    delta: torch.Tensor,
    beta: float = 0.01,
) -> torch.Tensor:
    """Compute the norm-aware scaling factor for Δz  (Eq. 11).

    ``scale = beta * ||hidden|| / (||delta|| + eps)``

    Returns the *scaled* perturbation ``delta * scale``, broadcastable to
    ``hidden``'s shape.
    """
    h_flat = hidden.detach().reshape(hidden.size(0), -1)
    d_flat = delta.detach().reshape(delta.size(0), -1)

    h_norm = h_flat.norm(dim=1)   # [B]
    d_norm = d_flat.norm(dim=1)   # [B]
    scale = beta * h_norm / (d_norm + 1e-7)  # [B]

    # Expand to hidden shape
    for _ in range(len(hidden.shape) - 1):
        scale = scale.unsqueeze(-1)

    return delta * scale


def update_perturbations(
    perturbations: Dict[int, torch.Tensor],
    gradients: Dict[int, torch.Tensor],
    active_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Apply gradient-descent update to Δz for the *active* layers only.

    For non-active layers the existing Δz is kept unchanged (detached &
    re-enabled for grad).

    Args:
        perturbations: current ``{layer_idx: delta_z}`` dict.
        gradients:     ``{layer_idx: grad_delta_z}`` from backward pass.
        active_layers: layer indices that were stochastically selected.

    Returns:
        Updated perturbations dict (all entries require grad).
    """
    updated = {}
    for idx, delta in perturbations.items():
        if idx in active_layers and idx in gradients:
            # Gradient *descent* on the loss we want to *maximise* →
            # subtract gradient (attack maximises target-class logit).
            new_delta = (delta - gradients[idx]).detach().requires_grad_(True)
            updated[idx] = new_delta
        else:
            # Keep the old perturbation, but detach & re-enable grad
            updated[idx] = delta.detach().requires_grad_(True)
    return updated
