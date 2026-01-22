import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from typing import Optional, List

def cross_entropy_loss(y_true, y_pred, device, reduction: str = "mean", class_weights: Optional[List[float]] = None):
    """
    Cross-entropy loss for multiclass classification (supports 2, 3, or more classes).
    If reduction == "none", returns per-sample losses (shape: [B]).
    This is useful when applying custom weighting, e.g. SQI-based weighting.
    
    Args:
        y_true: Ground truth labels (shape: [B])
        y_pred: Model predictions/logits (shape: [B, num_classes])
        device: Device to place tensors on
        reduction: Reduction mode ("none", "mean", or "sum")
        class_weights: Optional list of class weights for imbalanced datasets.
                      For binary classification: [weight_class_0, weight_class_1]
                      For 3 classes: [weight_class_0, weight_class_1, weight_class_2]
                      If None, all classes have equal weight.
    
    Returns:
        Loss tensor (scalar if reduction != "none", otherwise per-sample losses)
    """
    # Create loss function with optional class weights
    if class_weights is not None and len(class_weights) > 0:
        # Convert class_weights to tensor and move to device
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        # Verify that number of weights matches number of classes
        num_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 2
        if len(weights_tensor) != num_classes:
            raise ValueError(
                f"Number of class weights ({len(weights_tensor)}) does not match number of classes ({num_classes}). "
                f"Expected {num_classes} weights, got {len(weights_tensor)}."
            )
        loss_fn = nn.CrossEntropyLoss(weight=weights_tensor, reduction="none")
    else:
        # No class weights, use equal weights
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    loss = loss_fn(y_pred, y_true)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # "mean" or anything else falls back to mean
        return loss.mean()

# Keep backward compatibility alias
def binary_cross_entropy_with_logits(y_true, y_pred, device, reduction: str = "mean", class_weights: Optional[List[float]] = None):
    """
    Backward compatibility alias for cross_entropy_loss.
    Works for both binary and multiclass classification (2, 3, or more classes).
    
    Args:
        y_true: Ground truth labels (shape: [B])
        y_pred: Model predictions/logits (shape: [B, num_classes])
        device: Device to place tensors on
        reduction: Reduction mode ("none", "mean", or "sum")
        class_weights: Optional list of class weights for imbalanced datasets.
                      For binary classification: [weight_class_0, weight_class_1]
                      For 3 classes: [weight_class_0, weight_class_1, weight_class_2]
                      If None, all classes have equal weight.
    """
    return cross_entropy_loss(y_true, y_pred, device, reduction, class_weights)

loss_functions = {
    "bce": binary_cross_entropy_with_logits,  # Backward compatibility
    "cross_entropy": cross_entropy_loss,
    "ce": cross_entropy_loss,
}

def get_loss_function(loss_name):
    if loss_name in loss_functions:
        return loss_functions[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")