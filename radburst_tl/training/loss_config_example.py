"""
Simple loss configuration for train_utils.py compatibility
"""

def get_loss_config_for_scenario(scenario="balanced"):
    """
    Returns optimized loss configurations for different data scenarios.
    """
    
    configs = {
        "balanced": {
            "description": "Standard training with roughly balanced positive/negative samples",
            "loss_weights": {'focal': 0.8, 'iou': 1.0, 'boundary': 0.3},
            "focal_params": {'alpha': 0.75, 'gamma': 2.0},
        },
        
        "imbalanced": {
            "description": "Training with sparse positive samples (few radio bursts)",
            "loss_weights": {'focal': 1.2, 'iou': 1.5, 'boundary': 0.2},
            "focal_params": {'alpha': 0.8, 'gamma': 2.5},
        },
        
        "noisy": {
            "description": "Training data with significant noise and artifacts",
            "loss_weights": {'focal': 1.0, 'iou': 0.8, 'boundary': 0.5},
            "focal_params": {'alpha': 0.7, 'gamma': 2.5},
        },
        
        "boundary_critical": {
            "description": "High precision required for burst edge detection",
            "loss_weights": {'focal': 0.6, 'iou': 1.0, 'boundary': 0.8},
            "focal_params": {'alpha': 0.75, 'gamma': 2.0},
        },
        
        "original": {
            "description": "Use simple_combined_loss for backward compatibility",
            "use_simple_loss": True,
        }
    }
    
    return configs.get(scenario, configs["imbalanced"])
