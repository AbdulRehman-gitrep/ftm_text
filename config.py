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
        'target_margin': 0.0,    # Margin term for target-vs-non-target objective
        'change_push_weight': 0.0,  # Encourage movement from original embeddings

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
        'projection_swap_gap': 0.005,  # Allow near-tie swap away from original token

        # ── Semantic constraints ─────────────────────────────────
        'semantic_sim_threshold': 0.8,  # Min cosine similarity (sentence-transformers)
        'max_word_change_ratio': 0.3,   # Max fraction of words allowed to change

        # ── Evaluation ───────────────────────────────────────────
        'num_samples': 100,       # Number of samples to attack
        'seed': 42,

        'comment': 'Default settings for Text FTM attack (DistilBERT surrogate)',
    },

    2: {
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
        'num_iterations': 400,
        'alpha': 0.03,
        'target_margin': 0.1,
        'change_push_weight': 0.05,

        # ── FTM core hyperparameters ─────────────────────────────
        'ftm_beta': 0.05,
        'mix_prob': 0.2,
        'mix_upper_bound_feature': 0.6,
        'mix_lower_bound_feature': 0.0,

        # ── Transformer layer targeting ──────────────────────────
        'target_layers': [2, 3, 4, 5],

        # ── Feature mixing config ────────────────────────────────
        'mixed_image_type_feature': 'C',
        'shuffle_image_feature': 'None',
        'blending_mode_feature': 'M',
        'channelwise': False,

        # ── Word projection ──────────────────────────────────────
        'projection_freq': 100,
        'top_k_projection': 50,
        'projection_swap_gap': 0.015,

        # ── Semantic constraints ─────────────────────────────────
        'semantic_sim_threshold': 0.75,
        'max_word_change_ratio': 0.4,

        # ── Evaluation ───────────────────────────────────────────
        'num_samples': 100,
        'seed': 42,

        'comment': 'Balanced tuned settings: stronger updates with moderate constraints',
    },

    3: {
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
        'num_iterations': 500,
        'alpha': 0.05,
        'target_margin': 0.15,
        'change_push_weight': 0.1,

        # ── FTM core hyperparameters ─────────────────────────────
        'ftm_beta': 0.1,
        'mix_prob': 0.3,
        'mix_upper_bound_feature': 0.5,
        'mix_lower_bound_feature': 0.0,

        # ── Transformer layer targeting ──────────────────────────
        'target_layers': [2, 3, 4, 5],

        # ── Feature mixing config ────────────────────────────────
        'mixed_image_type_feature': 'C',
        'shuffle_image_feature': 'None',
        'blending_mode_feature': 'M',
        'channelwise': False,

        # ── Word projection ──────────────────────────────────────
        'projection_freq': 120,
        'top_k_projection': 50,
        'projection_swap_gap': 0.02,

        # ── Semantic constraints ─────────────────────────────────
        'semantic_sim_threshold': 0.7,
        'max_word_change_ratio': 0.5,

        # ── Evaluation ───────────────────────────────────────────
        'num_samples': 100,
        'seed': 42,

        'comment': 'Aggressive tuned settings: maximize attack strength for transfer',
    },

    4: {
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
        'num_iterations': 450,
        'alpha': 0.04,
        'target_margin': 0.12,
        'change_push_weight': 0.08,

        # ── FTM core hyperparameters ─────────────────────────────
        'ftm_beta': 0.08,
        'mix_prob': 0.25,
        'mix_upper_bound_feature': 0.55,
        'mix_lower_bound_feature': 0.05,

        # ── Transformer layer targeting ──────────────────────────
        'target_layers': [1, 2, 3, 4, 5],

        # ── Feature mixing config ────────────────────────────────
        'mixed_image_type_feature': 'C',
        'shuffle_image_feature': 'None',
        'blending_mode_feature': 'M',
        'channelwise': False,

        # ── Word projection ──────────────────────────────────────
        'projection_freq': 90,
        'top_k_projection': 50,
        'projection_swap_gap': 0.018,

        # ── Semantic constraints ─────────────────────────────────
        'semantic_sim_threshold': 0.72,
        'max_word_change_ratio': 0.45,

        # ── Evaluation ───────────────────────────────────────────
        'num_samples': 100,
        'seed': 42,

        'comment': 'Middle-ground tuned settings: stronger attack with controlled drift',
    },

    5: {
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
        'num_iterations': 500,
        'alpha': 0.06,
        'target_margin': 0.2,
        'change_push_weight': 0.15,

        # ── FTM core hyperparameters ─────────────────────────────
        'ftm_beta': 0.12,
        'mix_prob': 0.35,
        'mix_upper_bound_feature': 0.45,
        'mix_lower_bound_feature': 0.0,

        # ── Transformer layer targeting ──────────────────────────
        'target_layers': [1, 2, 3, 4, 5],

        # ── Feature mixing config ────────────────────────────────
        'mixed_image_type_feature': 'C',
        'shuffle_image_feature': 'None',
        'blending_mode_feature': 'M',
        'channelwise': False,

        # ── Word projection ──────────────────────────────────────
        'projection_freq': 140,
        'top_k_projection': 50,
        'projection_swap_gap': 0.03,

        # ── Semantic constraints ─────────────────────────────────
        'semantic_sim_threshold': 0.65,
        'max_word_change_ratio': 0.6,

        # ── Evaluation ───────────────────────────────────────────
        'num_samples': 100,
        'seed': 42,

        'comment': 'Exploratory high-strength settings for improved targeted/transfer ASR',
    },
}
