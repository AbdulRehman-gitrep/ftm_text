"""
Configuration for Text FTM Attack.
Mirrors the image FTM config structure, adapted for transformer-based text models.
"""

exp_configuration = {
    1: {
        # ── Dataset ──────────────────────────────────────────────
        'dataset': 'IMDB',
        'targeted': True,

        # ── Surrogate model ──────────────────────────────────────
        'surrogate_model': 'distilbert-base-uncased-finetuned-sst-2-english',

        # ── Black-box target models for transferability eval ─────
        'target_model_names': [
            'textattack/bert-base-uncased-imdb',
            'textattack/roberta-base-imdb',
            'textattack/xlnet-base-cased-imdb',
        ],

        # ── Attack iterations ────────────────────────────────────
        'num_iterations': 300,
        'alpha': 0.01,           # Step size for embedding updates (sign gradient)

        # ── FTM core hyperparameters ─────────────────────────────
        'ftm_beta': 0.01,        # Perturbation scaling factor (Eq. 11)
        'mix_prob': 0.1,         # Stochastic layer selection probability
        'mix_upper_bound_feature': 0.75,  # Max clean-feature mixing ratio
        'mix_lower_bound_feature': 0.0,   # Min clean-feature mixing ratio

        # ── Transformer layer targeting ──────────────────────────
        # DistilBERT has 6 transformer layers (0-5); target deeper half
        'target_layers': [3, 4, 5],

        # ── Feature mixing config ────────────────────────────────
        'mixed_image_type_feature': 'C',   # 'C': Clean features / 'A': Current batch
        'shuffle_image_feature': 'None',   # 'None' for text (single-sample attack)
        'blending_mode_feature': 'M',      # 'M': Linear interpolation
        'channelwise': False,              # No channel-wise mixing for text

        # ── Word projection ──────────────────────────────────────
        'projection_freq': 50,       # Project embeddings to words every N iters
        'top_k_projection': 50,      # Consider top-k nearest words during projection

        # ── Semantic constraints ─────────────────────────────────
        'semantic_sim_threshold': 0.8,  # Min cosine similarity (sentence-transformers)
        'max_word_change_ratio': 0.3,   # Max fraction of words allowed to change

        # ── Evaluation ───────────────────────────────────────────
        'num_samples': 100,       # Number of samples to attack
        'seed': 42,

        'comment': 'Default settings for Text FTM attack (DistilBERT surrogate)',
    },
}
