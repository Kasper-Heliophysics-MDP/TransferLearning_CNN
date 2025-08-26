"""
Interactive Loss Function Parameter Tuner for Solar Radio Burst Detection

This tool provides easy-to-use functions for tuning enhanced loss function parameters
based on validation performance.
"""

import torch
import numpy as np
from train_utils import combined_loss, simple_combined_loss
from hyperparameter_tuning_guide import get_tuning_strategy_by_problem, adaptive_parameter_adjustment


class LossTuner:
    """
    Interactive tool for tuning loss function parameters.
    """
    
    def __init__(self):
        self.history = []
        self.best_params = None
        self.best_metric = 0.0
        
    def diagnose_problem(self, val_metrics):
        """
        Diagnose the main training problem based on validation metrics.
        
        Args:
            val_metrics (dict): {'precision': float, 'recall': float, 'f1': float, 'iou': float}
            
        Returns:
            tuple: (problem_type, suggested_strategy)
        """
        precision = val_metrics.get('precision', 0.5)
        recall = val_metrics.get('recall', 0.5)
        f1 = val_metrics.get('f1', 0.5)
        iou = val_metrics.get('iou', 0.5)
        
        print(f"🔍 Diagnosing Training Performance:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")  
        print(f"  F1:        {f1:.3f}")
        print(f"  IoU:       {iou:.3f}")
        
        # Determine primary problem
        if recall < 0.6 and precision > 0.7:
            problem = "high_false_negatives"
            description = "🚨 Missing too many radio bursts (low recall)"
        elif precision < 0.6 and recall > 0.7:
            problem = "high_false_positives" 
            description = "🚨 Too many false detections (low precision)"
        elif f1 < 0.5:
            problem = "class_imbalance"
            description = "🚨 Overall poor performance (severe imbalance)"
        elif iou < 0.4:
            problem = "poor_boundaries"
            description = "🚨 Poor localization quality (low IoU)"
        else:
            problem = "balanced"
            description = "✅ Reasonable performance, minor tuning needed"
        
        print(f"\n{description}")
        
        strategies = get_tuning_strategy_by_problem()
        strategy = strategies.get(problem, strategies['class_imbalance'])
        
        return problem, strategy
    
    def suggest_parameters(self, val_metrics, current_params=None):
        """
        Suggest optimized parameters based on current performance.
        
        Args:
            val_metrics (dict): Current validation metrics
            current_params (dict): Current loss parameters (optional)
            
        Returns:
            dict: Suggested parameter configuration
        """
        if current_params is None:
            current_params = {
                'loss_weights': {'focal': 1.0, 'iou': 1.0, 'boundary': 0.2},
                'focal_params': {'alpha': 0.75, 'gamma': 2.0}
            }
        
        problem, strategy = self.diagnose_problem(val_metrics)
        
        print(f"\n🎯 Suggested Parameter Adjustments:")
        print(f"  Strategy: {strategy['problem']}")
        
        # Parse strategy adjustments into concrete parameters
        suggested = {
            'loss_weights': current_params['loss_weights'].copy(),
            'focal_params': current_params['focal_params'].copy()
        }
        
        adjustments = strategy['adjustments']
        
        # Apply focal parameter adjustments
        if 'focal_alpha' in adjustments:
            alpha_adj = adjustments['focal_alpha']
            if 'increase to' in alpha_adj:
                new_alpha = float(alpha_adj.split('to ')[1].split('-')[0])
                suggested['focal_params']['alpha'] = new_alpha
            elif 'decrease to' in alpha_adj:
                new_alpha = float(alpha_adj.split('to ')[1].split('-')[0])
                suggested['focal_params']['alpha'] = new_alpha
                
        if 'focal_gamma' in adjustments:
            gamma_adj = adjustments['focal_gamma']
            if 'increase to' in gamma_adj:
                new_gamma = float(gamma_adj.split('to ')[1].split('-')[0])
                suggested['focal_params']['gamma'] = new_gamma
            elif 'decrease to' in gamma_adj:
                new_gamma = float(gamma_adj.split('to ')[1].split('-')[0])
                suggested['focal_params']['gamma'] = new_gamma
        
        # Apply weight adjustments
        if 'focal_weight' in adjustments:
            weight_adj = adjustments['focal_weight']
            if 'increase to' in weight_adj:
                new_weight = float(weight_adj.split('to ')[1].split('-')[0])
                suggested['loss_weights']['focal'] = new_weight
            elif 'decrease to' in weight_adj:
                new_weight = float(weight_adj.split('to ')[1].split('-')[0])
                suggested['loss_weights']['focal'] = new_weight
                
        if 'iou_weight' in adjustments:
            weight_adj = adjustments['iou_weight']
            if 'increase to' in weight_adj:
                new_weight = float(weight_adj.split('to ')[1].split('-')[0])
                suggested['loss_weights']['iou'] = new_weight
                
        if 'boundary_weight' in adjustments:
            weight_adj = adjustments['boundary_weight']
            if 'increase to' in weight_adj:
                new_weight = float(weight_adj.split('to ')[1].split('-')[0])
                suggested['loss_weights']['boundary'] = new_weight
            elif 'keep low' in weight_adj:
                suggested['loss_weights']['boundary'] = 0.15
        
        print(f"\n  📋 Suggested Configuration:")
        print(f"    Loss weights: {suggested['loss_weights']}")
        print(f"    Focal params: {suggested['focal_params']}")
        print(f"\n  💡 Reasoning: {strategy['explanation']}")
        
        return suggested
    
    def generate_grid_search(self, center_params, grid_size="quick"):
        """
        Generate a grid search around given parameters.
        
        Args:
            center_params (dict): Center point for grid search
            grid_size (str): "quick" (8 configs), "medium" (27 configs), "full" (125 configs)
            
        Returns:
            list: List of parameter configurations to try
        """
        center_alpha = center_params['focal_params']['alpha']
        center_gamma = center_params['focal_params']['gamma']
        center_focal_w = center_params['loss_weights']['focal']
        center_iou_w = center_params['loss_weights']['iou']
        center_boundary_w = center_params['loss_weights']['boundary']
        
        if grid_size == "quick":
            # 2^3 = 8 configurations
            alpha_range = [center_alpha * 0.9, center_alpha * 1.1]
            gamma_range = [center_gamma * 0.8, center_gamma * 1.2] 
            focal_w_range = [center_focal_w * 0.8, center_focal_w * 1.2]
            iou_w_range = [center_iou_w]  # Keep fixed
            boundary_w_range = [center_boundary_w]  # Keep fixed
            
        elif grid_size == "medium":
            # 3^3 = 27 configurations
            alpha_range = [center_alpha * 0.85, center_alpha, center_alpha * 1.15]
            gamma_range = [center_gamma * 0.7, center_gamma, center_gamma * 1.3]
            focal_w_range = [center_focal_w * 0.7, center_focal_w, center_focal_w * 1.3]
            iou_w_range = [center_iou_w * 0.8, center_iou_w * 1.2]
            boundary_w_range = [center_boundary_w * 0.5, center_boundary_w * 2.0]
            
        else:  # full
            # 5^3 = 125 configurations
            alpha_range = np.linspace(center_alpha * 0.7, center_alpha * 1.3, 5)
            gamma_range = np.linspace(center_gamma * 0.5, center_gamma * 1.5, 5) 
            focal_w_range = np.linspace(center_focal_w * 0.5, center_focal_w * 1.5, 5)
            iou_w_range = np.linspace(center_iou_w * 0.5, center_iou_w * 1.5, 5)
            boundary_w_range = np.linspace(center_boundary_w * 0.5, center_boundary_w * 2.0, 5)
        
        configurations = []
        
        for alpha in alpha_range:
            for gamma in gamma_range:
                for focal_w in focal_w_range:
                    for iou_w in iou_w_range:
                        for boundary_w in boundary_w_range:
                            config = {
                                'loss_weights': {
                                    'focal': float(focal_w),
                                    'iou': float(iou_w),
                                    'boundary': float(boundary_w)
                                },
                                'focal_params': {
                                    'alpha': float(alpha),
                                    'gamma': float(gamma)
                                }
                            }
                            configurations.append(config)
        
        print(f"🔍 Generated {len(configurations)} configurations for {grid_size} grid search")
        return configurations
    
    def record_experiment(self, params, metrics):
        """
        Record an experiment result for tracking progress.
        
        Args:
            params (dict): Parameters used
            metrics (dict): Resulting validation metrics
        """
        experiment = {
            'params': params.copy(),
            'metrics': metrics.copy(),
            'f1': metrics.get('f1', 0.0),
            'iou': metrics.get('iou', 0.0)
        }
        
        self.history.append(experiment)
        
        # Update best if this is better (using F1 as primary metric)
        if experiment['f1'] > self.best_metric:
            self.best_metric = experiment['f1']
            self.best_params = params.copy()
            print(f"🏆 New best F1: {self.best_metric:.3f}")
        
    def get_best_params(self):
        """
        Return the best parameters found so far.
        """
        if self.best_params is None:
            print("❌ No experiments recorded yet")
            return None
            
        print(f"🏆 Best Parameters (F1: {self.best_metric:.3f}):")
        print(f"  Loss weights: {self.best_params['loss_weights']}")
        print(f"  Focal params: {self.best_params['focal_params']}")
        
        return self.best_params
    
    def compare_with_baseline(self, y_true, y_pred_enhanced, y_pred_baseline=None):
        """
        Compare enhanced loss results with baseline.
        
        Args:
            y_true (torch.Tensor): Ground truth
            y_pred_enhanced (torch.Tensor): Predictions with enhanced loss
            y_pred_baseline (torch.Tensor): Predictions with original loss (optional)
        """
        print(f"📊 Enhanced vs Baseline Comparison:")
        print(f"=" * 40)
        
        # Enhanced metrics
        enhanced_metrics = self._compute_metrics(y_true, y_pred_enhanced)
        print(f"Enhanced Loss:")
        print(f"  Precision: {enhanced_metrics['precision']:.3f}")
        print(f"  Recall:    {enhanced_metrics['recall']:.3f}")
        print(f"  F1:        {enhanced_metrics['f1']:.3f}")
        print(f"  IoU:       {enhanced_metrics['iou']:.3f}")
        
        if y_pred_baseline is not None:
            baseline_metrics = self._compute_metrics(y_true, y_pred_baseline)
            print(f"\nBaseline Loss:")
            print(f"  Precision: {baseline_metrics['precision']:.3f}")
            print(f"  Recall:    {baseline_metrics['recall']:.3f}")
            print(f"  F1:        {baseline_metrics['f1']:.3f}")
            print(f"  IoU:       {baseline_metrics['iou']:.3f}")
            
            # Improvement
            f1_improvement = enhanced_metrics['f1'] - baseline_metrics['f1']
            iou_improvement = enhanced_metrics['iou'] - baseline_metrics['iou']
            
            print(f"\n🎯 Improvement:")
            print(f"  F1:  {f1_improvement:+.3f} ({f1_improvement/baseline_metrics['f1']*100:+.1f}%)")
            print(f"  IoU: {iou_improvement:+.3f} ({iou_improvement/baseline_metrics['iou']*100:+.1f}%)")
    
    def _compute_metrics(self, y_true, y_pred, threshold=0.5):
        """Helper function to compute metrics."""
        y_pred_binary = (y_pred > threshold).float()
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_binary.flatten()
        
        tp = (y_true_flat * y_pred_flat).sum().item()
        fp = ((1 - y_true_flat) * y_pred_flat).sum().item()
        fn = (y_true_flat * (1 - y_pred_flat)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-8)
        
        return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou}


# =============================================================================
# QUICK START FUNCTIONS
# =============================================================================

def quick_tune(val_metrics):
    """
    Quick parameter tuning based on validation metrics.
    
    Args:
        val_metrics (dict): {'precision': float, 'recall': float, 'f1': float, 'iou': float}
        
    Returns:
        dict: Suggested parameters ready to use
    """
    tuner = LossTuner()
    suggested = tuner.suggest_parameters(val_metrics)
    
    print(f"\n🚀 Ready to use configuration:")
    print(f"loss = combined_loss(")
    print(f"    y_true, y_pred,")
    print(f"    loss_weights={suggested['loss_weights']},")
    print(f"    focal_params={suggested['focal_params']}")
    print(f")")
    
    return suggested


def print_tuning_checklist():
    """
    Print a quick checklist for systematic tuning.
    """
    checklist = [
        "✅ Record baseline performance with original loss",
        "✅ Identify primary problem (precision vs recall)",
        "✅ Apply suggested parameter adjustments",
        "✅ Train for 10-20 epochs to see trend",
        "✅ If improved, continue; if not, try different approach",
        "✅ Consider grid search for fine-tuning",
        "✅ Validate final model on test set"
    ]
    
    print("📋 Hyperparameter Tuning Checklist:")
    for item in checklist:
        print(f"  {item}")


if __name__ == "__main__":
    # Example usage
    print("🎯 Loss Function Tuner - Example Usage")
    print("=" * 50)
    
    # Simulate validation metrics
    example_metrics = {
        'precision': 0.82,
        'recall': 0.48,
        'f1': 0.61,
        'iou': 0.44
    }
    
    # Quick tuning
    suggested = quick_tune(example_metrics)
    
    print("\n" + "="*50)
    print_tuning_checklist()
