"""
Black-box target models for transferability evaluation.
These models are *never* accessed during attack generation — only for measuring
whether adversarial texts transfer across architectures.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


class BlackBoxModel:
    """Wraps a HuggingFace classifier for black-box evaluation."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Determine label mapping from the model's config
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> int:
        """Return predicted class index for a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).item()

    def predict_batch(self, texts: list[str]) -> list[int]:
        """Return predicted class indices for a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).tolist()

    def __repr__(self):
        return f"BlackBoxModel({self.model_name})"


def load_black_box_models(
    model_names: list[str], device: str = "cpu"
) -> list[BlackBoxModel]:
    """Load multiple black-box models for evaluation."""
    models = []
    for name in model_names:
        print(f"  Loading black-box model: {name}")
        models.append(BlackBoxModel(name, device))
    return models
