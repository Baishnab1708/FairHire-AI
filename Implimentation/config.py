# Configuration for debiased resume ranking system

# Data settings
DATA_CONFIG = {
    'resumes_dir': 'data/resumes',
    'bias_patterns_dir': 'data/bias_patterns',
    'sample_size': 100  # Number of sample resumes to generate if no data is found
}

# Model settings
MODEL_CONFIG = {
    'primary_ai': {
        'max_features': 1000,  # Maximum number of features for vectorizer
        'model_type': 'logistic_regression'  # Model type for primary AI
    },
    'adversarial_ai': {
        'sensitive_categories': [
            'gender', 
            'race', 
            'age', 
            'education'
        ],
        'bias_threshold': 0.2  # Threshold for bias detection
    },
    'unlearning': {
        'remove_biased_features': True  # Whether to completely remove biased features
    }
}

# Evaluation settings
EVAL_CONFIG = {
    'test_split': 0.2,  # Fraction of data to use for testing
    'metrics': [
        'mean_score_change',
        'rank_change',
        'bias_reduction'
    ]
}