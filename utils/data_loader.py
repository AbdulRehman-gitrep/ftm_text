"""
Data loader for IMDB sentiment classification dataset.
Downloads via HuggingFace `datasets` library and prepares
(text, true_label, target_label) tuples for the attack.
"""

from typing import List, Tuple
import random


def load_imdb_dataset(
    num_samples: int = 100,
    seed: int = 42,
    split: str = "test",
) -> List[Tuple[str, int, int]]:
    """Load IMDB review samples for adversarial attack.

    Each sample is ``(text, true_label, target_label)`` where
    ``target_label = 1 - true_label`` (flip sentiment).

    IMDB labels: 0 = Negative, 1 = Positive.
    DistilBERT-SST2 labels: 0 = NEGATIVE, 1 = POSITIVE — compatible.

    Args:
        num_samples: how many samples to return.
        seed:        random seed for reproducible selection.
        split:       'test' or 'train'.

    Returns:
        List of ``(text, true_label, target_label)`` tuples.
    """
    from datasets import load_dataset

    print(f"Loading IMDB dataset ({split} split) ...")
    dataset = load_dataset("imdb", split=split)

    # Shuffle deterministically
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    samples = []
    for i in indices:
        if len(samples) >= num_samples:
            break

        text = dataset[i]["text"]
        true_label = dataset[i]["label"]  # 0=neg, 1=pos

        # Skip very long reviews (>256 words) to keep attack fast
        if len(text.split()) > 256:
            continue

        target_label = 1 - true_label  # flip sentiment
        samples.append((text, true_label, target_label))

    print(f"Loaded {len(samples)} samples (requested {num_samples})")
    return samples


def load_custom_csv(
    csv_path: str,
    text_col: str = "review",
    label_col: str = "sentiment",
    num_samples: int = 100,
    seed: int = 42,
) -> List[Tuple[str, int, int]]:
    """Load samples from a custom CSV file.

    Expected columns:
        - ``text_col``: raw text
        - ``label_col``: 'positive'/'negative' or 0/1

    Returns:
        List of ``(text, true_label, target_label)`` tuples.
    """
    import csv

    label_map = {"positive": 1, "negative": 0, "pos": 1, "neg": 0}
    samples_raw = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row[text_col].strip()
            raw_label = row[label_col].strip().lower()
            if raw_label in label_map:
                true_label = label_map[raw_label]
            else:
                true_label = int(raw_label)
            samples_raw.append((text, true_label))

    rng = random.Random(seed)
    rng.shuffle(samples_raw)
    samples_raw = samples_raw[:num_samples]

    samples = []
    for text, true_label in samples_raw:
        target_label = 1 - true_label
        samples.append((text, true_label, target_label))

    print(f"Loaded {len(samples)} samples from {csv_path}")
    return samples
