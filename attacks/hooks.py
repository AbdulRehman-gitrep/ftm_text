"""
PyTorch forward hooks for transformer layers — the core of FTM.

Mirrors the image `FeatureTuning` class:
  • Record mode  → store clean hidden states, initialise zero Δz
  • Attack mode  → stochastically add scaled Δz and mix with clean features
"""

import torch
import torch.nn as nn
import random
from typing import Callable, Dict, List, Optional


class TextFeatureTuning(nn.Module):
    """
    Wraps a HuggingFace text model and registers forward hooks on specified
    transformer layers to perform feature-level perturbation and mixup.

    Analogous to ``FeatureTuning`` in the image FTM ``attacks.py``.
    """

    def __init__(
        self,
        surrogate_model,          # SurrogateModel instance
        exp_settings: dict,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.exp_settings = exp_settings

        self.model = surrogate_model.model
        self.tokenizer = surrogate_model.tokenizer

        # Hyper-parameters
        self.target_layer_indices: List[int] = exp_settings["target_layers"]
        self.prob = exp_settings["mix_prob"]           # 0.1
        self.beta = exp_settings["ftm_beta"]           # 0.01
        self.mix_upper = exp_settings["mix_upper_bound_feature"]  # 0.75
        self.mix_lower = exp_settings["mix_lower_bound_feature"]  # 0.0
        self.blending_mode = exp_settings["blending_mode_feature"]  # 'M'

        # Internal state
        self.record = False
        self.clean_features: Dict[int, torch.Tensor] = {}
        self.perturbations: Dict[int, torch.Tensor] = {}   # Δz per layer
        self.mixing_triggered: Dict[int, bool] = {}
        self.forward_hooks = []

        # Register hooks on target transformer layers
        transformer_layers = surrogate_model.get_transformer_layers()
        for idx in self.target_layer_indices:
            layer = transformer_layers[idx]
            handle = layer.register_forward_hook(self._make_hook(idx))
            self.forward_hooks.append(handle)

    # ── Hook factory ─────────────────────────────────────────────────

    def _make_hook(self, layer_idx: int) -> Callable:
        """Return a hook closure for *layer_idx*."""

        def hook_fn(module, input, output):
            # DistilBERT transformer layer returns a tuple: (hidden_states,)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # ── Record mode ──────────────────────────────────────
            if self.record:
                self.clean_features[layer_idx] = hidden.clone().detach()
                self.perturbations[layer_idx] = (
                    torch.zeros_like(hidden).detach().requires_grad_(True)
                )
                self.mixing_triggered[layer_idx] = False
                return  # keep original output unchanged

            # ── Attack mode (only if we already have clean features) ──
            if layer_idx not in self.clean_features:
                return  # layer wasn't recorded — pass through

            # Stochastic layer selection
            triggered = random.random() <= self.prob
            self.mixing_triggered[layer_idx] = triggered

            # --- Always apply Δz perturbation (Eq. 11) ---
            hidden_flat = hidden.detach().reshape(hidden.size(0), -1)
            delta_flat = self.perturbations[layer_idx].detach().reshape(
                hidden.size(0), -1
            )
            h_norm = hidden_flat.norm(dim=1)          # [B]
            d_norm = delta_flat.norm(dim=1)            # [B]
            scale = self.beta * h_norm / (d_norm + 1e-7)  # [B]

            # Expand scale to match hidden shape
            for _ in range(len(hidden.shape) - 1):
                scale = scale.unsqueeze(-1)

            perturbed = hidden + self.perturbations[layer_idx] * scale

            if triggered:
                # ── Clean feature mixing (Eq. 12) ────────────────
                clean = self.clean_features[layer_idx]

                mix_range = self.mix_upper - self.mix_lower
                alpha = random.random() * mix_range + self.mix_lower  # scalar

                if self.blending_mode == "M":  # convex interpolation
                    mixed = (1 - alpha) * perturbed + alpha * clean
                else:  # 'A' — additive
                    mixed = perturbed + alpha * clean

                if isinstance(output, tuple):
                    return (mixed,) + output[1:]
                return mixed

            # Non-triggered: just use perturbed
            if isinstance(output, tuple):
                return (perturbed,) + output[1:]
            return perturbed

        return hook_fn

    # ── Mode toggles ─────────────────────────────────────────────────

    def start_feature_record(self):
        """Switch to recording mode (first forward pass stores clean features)."""
        self.record = True

    def end_feature_record(self):
        """Switch back to attack mode."""
        self.record = False

    # ── Forward pass ─────────────────────────────────────────────────

    def forward_from_embeddings(self, embeddings, attention_mask):
        """Run model from embeddings (hooks fire automatically)."""
        self.mixing_triggered = {}  # reset per forward pass
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask)

    def forward(self, **inputs):
        """Standard forward pass (hooks fire automatically)."""
        self.mixing_triggered = {}
        return self.model(**inputs)

    # ── Cleanup ──────────────────────────────────────────────────────

    def remove_hooks(self):
        """Remove all registered hooks and free buffers."""
        for h in self.forward_hooks:
            h.remove()
        self.forward_hooks.clear()
        self.clean_features.clear()
        self.perturbations.clear()
        self.mixing_triggered.clear()
