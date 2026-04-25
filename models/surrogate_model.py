"""
Surrogate model wrapper for the text FTM attack.
Uses DistilBERT fine-tuned on SST-2 (sentiment analysis) as the white-box
surrogate — analogous to ResNet50 in the image FTM attack.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class SurrogateModel:
    """Wraps a HuggingFace text classifier for use with the FTM attack."""

    # SST-2 label mapping (DistilBERT-SST2 uses 0=NEGATIVE, 1=POSITIVE)
    LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ── Convenience accessors ────────────────────────────────────────

    def get_embedding_layer(self):
        """Return the nn.Embedding layer (word embeddings)."""
        return self.model.get_input_embeddings()

    def get_vocab_embeddings(self):
        """Return the full vocabulary embedding weight matrix [vocab, hidden]."""
        return self.get_embedding_layer().weight

    def get_transformer_layers(self):
        """Return the list of transformer blocks."""
        # DistilBERT: model.distilbert.transformer.layer
        if hasattr(self.model, "distilbert"):
            return self.model.distilbert.transformer.layer
        # BERT-style: model.bert.encoder.layer
        if hasattr(self.model, "bert"):
            return self.model.bert.encoder.layer
        raise AttributeError(
            f"Cannot find transformer layers for model {self.model_name}"
        )

    @property
    def num_layers(self) -> int:
        return len(self.get_transformer_layers())

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    # ── Tokenisation ─────────────────────────────────────────────────

    def tokenize(self, text: str):
        """Tokenize a single text string → dict of tensors on self.device."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def tokens_to_text(self, input_ids):
        """Decode token IDs back to a string, skipping special tokens."""
        return self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

    # ── Forward helpers ──────────────────────────────────────────────

    def forward_from_embeddings(self, embeddings, attention_mask):
        """Run model from embeddings (bypassing the lookup table)."""
        return self.model(inputs_embeds=embeddings, attention_mask=attention_mask)

    def predict(self, text: str):
        """Quick helper: text → predicted label index."""
        inputs = self.tokenize(text)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).item()
