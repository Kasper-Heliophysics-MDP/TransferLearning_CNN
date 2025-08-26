"""
Enhanced Loss Function Hyperparameter Tuning Guide for Solar Radio Burst Detection

This guide provides comprehensive strategies for tuning the enhanced loss function parameters
based on data characteristics and validation performance.
"""

import numpy as np
import torch
from train_utils import combined_loss, focal_loss, boundary_loss, adaptive_iou_loss


# =============================================================================
# PARAMETER OVERVIEW
# =============================================================================

def get_tunable_parameters():
    """
    Returns all tunable parameters with their effects and typical ranges.
    """
    parameters = {
        "loss_weights": {
            "focal": {
                "range": [0.5, 2.0],
                "default": 1.0,
                "effect": "Controls class imbalance handling strength",
                "increase_when": "High false negative rate (missing bursts)",
                "decrease_when": "High false positive rate (too much noise)"
            },
            "iou": {
                "range": [0.5, 2.0], 
                "default": 1.0,
                "effect": "Controls overlap quality emphasis",
                "increase_when": "Poor localization accuracy",
                "decrease_when": "Need to focus more on classification"
            },
            "boundary": {
                "range": [0.1, 1.0],
                "default": 0.2,
                "effect": "Controls edge detection precision",
                "increase_when": "Blurry or imprecise burst boundaries",
                "decrease_when": "Boundary loss dominates too much"
            }
        },
        "focal_params": {
            "alpha": {
                "range": [0.25, 0.95],
                "default": 0.75,
                "effect": "Balances positive vs negative class importance",
                "increase_when": "Missing too many radio bursts (favor recall)",
                "decrease_when": "Too many false positives (favor precision)"
            },
            "gamma": {
                "range": [1.0, 5.0],
                "default": 2.0,
                "effect": "Controls focus on hard examples",
                "increase_when": "Model struggles with difficult cases",
                "decrease_when": "Training becomes unstable"
            }
        }
    }
    return parameters


# =============================================================================
# SYSTEMATIC TUNING STRATEGIES
# =============================================================================

def get_tuning_strategy_by_problem():
    """
    Returns specific tuning strategies based on observed training problems.
    """
    strategies = {
        "high_false_negatives": {
            "problem": "Model missing many radio bursts",
            "symptoms": ["Low recall", "High precision", "Missing small/weak bursts"],
            "adjustments": {
                "focal_alpha": "increase to 0.8-0.9",
                "focal_gamma": "increase to 3.0-4.0", 
                "focal_weight": "increase to 1.5-2.0",
                "iou_weight": "increase to 1.5",
                "boundary_weight": "keep low 0.1-0.2"
            },
            "explanation": "Focus heavily on positive class and hard examples"
        },
        
        "high_false_positives": {
            "problem": "Model detecting too much noise as bursts",
            "symptoms": ["High recall", "Low precision", "Noisy predictions"],
            "adjustments": {
                "focal_alpha": "decrease to 0.6-0.7",
                "focal_gamma": "keep moderate 2.0-2.5",
                "focal_weight": "decrease to 0.8-1.0", 
                "iou_weight": "increase to 1.5",
                "boundary_weight": "increase to 0.3-0.5"
            },
            "explanation": "Focus on precision and boundary quality"
        },
        
        "poor_boundaries": {
            "problem": "Burst boundaries are imprecise or blurry",
            "symptoms": ["Good detection", "Poor edge quality", "Fuzzy masks"],
            "adjustments": {
                "boundary_weight": "increase to 0.5-0.8",
                "focal_weight": "keep moderate 1.0",
                "iou_weight": "keep moderate 1.0",
                "focal_alpha": "keep default 0.75",
                "focal_gamma": "keep default 2.0"
            },
            "explanation": "Emphasize edge detection without disrupting classification"
        },
        
        "training_instability": {
            "problem": "Loss oscillates or training doesn't converge",
            "symptoms": ["Erratic loss curves", "Poor convergence", "Gradient issues"],
            "adjustments": {
                "focal_gamma": "decrease to 1.5-2.0",
                "boundary_weight": "decrease to 0.1-0.2",
                "all_weights": "scale down proportionally",
                "focal_alpha": "move toward 0.5 (balanced)"
            },
            "explanation": "Reduce aggressive focusing and boundary emphasis"
        },
        
        "class_imbalance": {
            "problem": "Severe class imbalance (very few radio bursts)",
            "symptoms": ["Model predicts mostly background", "Low recall"],
            "adjustments": {
                "focal_alpha": "increase to 0.85-0.95",
                "focal_gamma": "increase to 3.0-5.0",
                "focal_weight": "increase to 1.5-2.0",
                "iou_weight": "increase to 1.5",
                "boundary_weight": "keep low 0.1"
            },
            "explanation": "Heavily focus on rare positive class"
        }
    }
    return strategies


# =============================================================================
# GRID SEARCH CONFIGURATIONS
# =============================================================================

def get_grid_search_configs():
    """
    Returns pre-defined configurations for systematic grid search.
    """
    # Conservative grid (fewer experiments, safer ranges)
    conservative_grid = {
        "focal_alpha": [0.7, 0.75, 0.8],
        "focal_gamma": [2.0, 2.5, 3.0],
        "focal_weight": [1.0, 1.2, 1.5],
        "iou_weight": [1.0, 1.2, 1.5], 
        "boundary_weight": [0.1, 0.2, 0.3]
    }
    
    # Aggressive grid (more experiments, wider ranges)
    aggressive_grid = {
        "focal_alpha": [0.6, 0.7, 0.75, 0.8, 0.85],
        "focal_gamma": [1.5, 2.0, 2.5, 3.0, 4.0],
        "focal_weight": [0.8, 1.0, 1.2, 1.5, 2.0],
        "iou_weight": [0.8, 1.0, 1.2, 1.5, 2.0],
        "boundary_weight": [0.1, 0.2, 0.3, 0.5, 0.8]
    }
    
    # Quick validation grid (for fast iteration)
    quick_grid = {
        "focal_alpha": [0.75, 0.8],
        "focal_gamma": [2.0, 3.0], 
        "focal_weight": [1.0, 1.5],
        "iou_weight": [1.0, 1.5],
        "boundary_weight": [0.2, 0.3]
    }
    
    return {
        "conservative": conservative_grid,
        "aggressive": aggressive_grid, 
        "quick": quick_grid
    }


# =============================================================================
# ADAPTIVE TUNING STRATEGIES
# =============================================================================

def adaptive_parameter_adjustment(val_metrics, current_params, adjustment_rate=0.1):
    """
    Automatically adjust parameters based on validation metrics.
    
    Args:
        val_metrics (dict): {'precision': float, 'recall': float, 'f1': float, 'iou': float}
        current_params (dict): Current loss function parameters
        adjustment_rate (float): How aggressively to adjust (0.1 = 10% changes)
    
    Returns:
        dict: Suggested parameter adjustments
    """
    suggestions = {}
    
    precision = val_metrics.get('precision', 0.5)
    recall = val_metrics.get('recall', 0.5)
    f1 = val_metrics.get('f1', 0.5)
    iou = val_metrics.get('iou', 0.5)
    
    # Precision vs Recall trade-off
    if recall < 0.6 and precision > 0.8:  # Missing bursts
        suggestions['focal_alpha'] = min(0.9, current_params.get('focal_alpha', 0.75) + adjustment_rate)
        suggestions['focal_gamma'] = min(4.0, current_params.get('focal_gamma', 2.0) + 0.5)
        suggestions['focal_weight'] = min(2.0, current_params.get('focal_weight', 1.0) + adjustment_rate)
        
    elif precision < 0.6 and recall > 0.8:  # Too many false positives
        suggestions['focal_alpha'] = max(0.5, current_params.get('focal_alpha', 0.75) - adjustment_rate)
        suggestions['boundary_weight'] = min(0.5, current_params.get('boundary_weight', 0.2) + adjustment_rate)
    
    # IoU-specific adjustments
    if iou < 0.5:  # Poor overlap
        suggestions['iou_weight'] = min(2.0, current_params.get('iou_weight', 1.0) + adjustment_rate)
        
    # F1-specific adjustments  
    if f1 < 0.6:  # Overall poor performance
        suggestions['focal_weight'] = min(1.5, current_params.get('focal_weight', 1.0) + adjustment_rate)
    
    return suggestions


# =============================================================================
# VALIDATION METRICS FOR TUNING
# =============================================================================

def compute_detailed_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute detailed metrics for hyperparameter tuning decisions.
    
    Args:
        y_true (torch.Tensor): Ground truth masks
        y_pred (torch.Tensor): Predicted probability masks
        threshold (float): Binary threshold
        
    Returns:
        dict: Comprehensive metrics for tuning decisions
    """
    # Convert to binary predictions
    y_pred_binary = (y_pred > threshold).float()
    
    # Flatten for computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Basic metrics
    tp = (y_true_flat * y_pred_flat).sum().item()
    fp = ((1 - y_true_flat) * y_pred_flat).sum().item()
    fn = (y_true_flat * (1 - y_pred_flat)).sum().item()
    tn = ((1 - y_true_flat) * (1 - y_pred_flat)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)
    
    # Class distribution
    pos_ratio = y_true_flat.mean().item()
    pred_pos_ratio = y_pred_flat.mean().item()
    
    # Confidence distribution
    high_conf_pixels = (y_pred > 0.8).sum().item()
    low_conf_pixels = (y_pred < 0.2).sum().item()
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'pos_ratio': pos_ratio,
        'pred_pos_ratio': pred_pos_ratio,
        'high_conf_pixels': high_conf_pixels,
        'low_conf_pixels': low_conf_pixels,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }
    
    return metrics


# =============================================================================
# PRACTICAL TUNING WORKFLOW
# =============================================================================

def get_tuning_workflow():
    """
    Returns a step-by-step tuning workflow.
    """
    workflow = {
        "step_1": {
            "title": "Baseline Establishment",
            "actions": [
                "Train with default parameters (alpha=0.75, gamma=2.0, weights=1.0,1.0,0.2)",
                "Record validation metrics (precision, recall, F1, IoU)",
                "Identify primary problem (false negatives/positives/boundaries)"
            ]
        },
        
        "step_2": {
            "title": "Problem-Specific Adjustment",
            "actions": [
                "Use get_tuning_strategy_by_problem() to get targeted adjustments",
                "Make conservative changes (±10-20% from default)",
                "Train for 10-20 epochs to see trend"
            ]
        },
        
        "step_3": {
            "title": "Fine-tuning",
            "actions": [
                "If improvement seen, continue in same direction",
                "If no improvement, try different parameter combination",
                "Monitor individual loss components (focal, iou, boundary)"
            ]
        },
        
        "step_4": {
            "title": "Grid Search (Optional)",
            "actions": [
                "Use quick_grid for 2x2x2 = 8 combinations",
                "Train each for 20-30 epochs",
                "Select best based on validation F1 or IoU"
            ]
        },
        
        "step_5": {
            "title": "Final Validation",
            "actions": [
                "Train best configuration for full epochs",
                "Compare with original loss function",
                "Validate on test set"
            ]
        }
    }
    
    return workflow


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_tuning_session():
    """
    Example of a complete tuning session.
    """
    print("🎯 Hyperparameter Tuning Example Session")
    print("=" * 50)
    
    # Step 1: Analyze current problem
    print("\n📊 Step 1: Problem Analysis")
    current_metrics = {
        'precision': 0.85,
        'recall': 0.45,    # Low recall = missing bursts
        'f1': 0.59,
        'iou': 0.42
    }
    
    print(f"Current metrics: {current_metrics}")
    print("🔍 Diagnosis: High precision, low recall → Missing radio bursts")
    
    # Step 2: Get strategy
    strategies = get_tuning_strategy_by_problem()
    strategy = strategies['high_false_negatives']
    
    print(f"\n🎯 Step 2: Strategy Selection")
    print(f"Problem: {strategy['problem']}")
    print(f"Adjustments:")
    for param, value in strategy['adjustments'].items():
        print(f"  {param}: {value}")
    
    # Step 3: Suggested parameters
    print(f"\n🔧 Step 3: Suggested Configuration")
    suggested_config = {
        'loss_weights': {'focal': 1.5, 'iou': 1.5, 'boundary': 0.2},
        'focal_params': {'alpha': 0.85, 'gamma': 3.0}
    }
    print(f"Suggested config: {suggested_config}")
    
    # Step 4: Grid search options
    print(f"\n🔍 Step 4: Alternative Grid Search")
    grids = get_grid_search_configs()
    print(f"Quick grid combinations: {len(grids['quick']['focal_alpha']) * len(grids['quick']['focal_gamma'])}")
    
    return suggested_config


if __name__ == "__main__":
    # Run example
    config = example_tuning_session()
    
    print(f"\n🚀 Ready to tune! Start with the suggested configuration above.")
    print(f"\n💡 Key tuning tips:")
    print(f"  1. Change one parameter group at a time")
    print(f"  2. Make small changes (10-20%) initially") 
    print(f"  3. Monitor validation metrics every 5-10 epochs")
    print(f"  4. Focus on your primary metric (F1 or IoU)")
    print(f"  5. Don't over-tune on validation set!")
