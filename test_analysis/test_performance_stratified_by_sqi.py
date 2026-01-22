"""
Script to analyze test performance stratified by SQI (Signal Quality Index).
Loads saved test data and creates visualizations showing how metrics vary with SQI.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchmetrics
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict
import seaborn as sns

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_test_data(data_path: str) -> Dict[str, np.ndarray]:
    """
    Load test data from saved .npz file.
    
    Args:
        data_path: Path to the .npz file containing test data
        
    Returns:
        Dictionary with keys: 'predictions', 'targets', 'logits', 'sqi'
    """
    data = np.load(data_path)
    return {
        'predictions': data['predictions'],
        'targets': data['targets'],
        'logits': data['logits'],
        'sqi': data['sqi']
    }


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   logits: torch.Tensor, task: str = "multiclass", 
                   num_classes: int = 3) -> Dict[str, float]:
    """
    Compute various metrics given predictions, targets, and logits.
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        logits: Model logits
        task: Task type ("binary" or "multiclass")
        num_classes: Number of classes
        
    Returns:
        Dictionary of metric names and values
    """
    device = predictions.device
    metrics = {}
    
    # Accuracy
    acc_metric = torchmetrics.Accuracy(
        task, num_classes=num_classes, average="macro"
    ).to(device)
    acc_metric.update(predictions, targets)
    metrics['accuracy'] = acc_metric.compute().item()
    
    # F1 Score
    f1_metric = torchmetrics.F1Score(
        task, num_classes=num_classes, average="macro"
    ).to(device)
    f1_metric.update(predictions, targets)
    metrics['f1'] = f1_metric.compute().item()
    
    # Sensitivity (Recall)
    if task == "binary":
        sens_metric = torchmetrics.Recall(task="binary").to(device)
    else:
        sens_metric = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
    sens_metric.update(predictions, targets)
    metrics['sensitivity'] = sens_metric.compute().item()
    
    # PPV (Precision)
    if task == "binary":
        ppv_metric = torchmetrics.Precision(task="binary").to(device)
    else:
        ppv_metric = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
    ppv_metric.update(predictions, targets)
    metrics['ppv'] = ppv_metric.compute().item()
    
    # Specificity
    spec_metric = torchmetrics.Specificity(
        task, num_classes=num_classes, average="macro"
    ).to(device)
    spec_metric.update(predictions, targets)
    metrics['specificity'] = spec_metric.compute().item()
    
    # AUC
    if task == "binary":
        auc_logits = logits[:, 1]  # Take logit for positive class
        auc_metric = torchmetrics.AUROC(task="binary").to(device)
    else:
        auc_logits = logits
        auc_metric = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
    auc_metric.update(auc_logits, targets)
    metrics['auc'] = auc_metric.compute().item()
    
    return metrics


def compute_stratified_metrics(data: Dict[str, np.ndarray], 
                               sqi_bins: np.ndarray = None,
                               task: str = "multiclass",
                               num_classes: int = 3) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Compute metrics for each SQI bin.
    
    Args:
        data: Dictionary containing test data
        sqi_bins: SQI bin edges. If None, uses default bins: [0, 10, 20, ..., 100]
        task: Task type ("binary" or "multiclass")
        num_classes: Number of classes
        
    Returns:
        Tuple of (bin_centers, metrics_dict, sample_counts)
        - bin_centers: Center values of each SQI bin
        - metrics_dict: Dictionary with metric names as keys and arrays of values per bin
        - sample_counts: Number of samples in each bin
    """
    predictions = torch.from_numpy(data['predictions'])
    targets = torch.from_numpy(data['targets'])
    logits = torch.from_numpy(data['logits'])
    sqi = data['sqi']
    
    # Create bins if not provided
    if sqi_bins is None:
        sqi_bins = np.arange(0, 101, 10)  # 0-10, 10-20, ..., 90-100
    
    # Compute bin centers
    bin_centers = (sqi_bins[:-1] + sqi_bins[1:]) / 2
    
    # Initialize metric arrays
    metrics_dict = {
        'accuracy': [],
        'f1': [],
        'sensitivity': [],
        'ppv': [],
        'specificity': [],
        'auc': []
    }
    sample_counts = []
    
    # Compute metrics for each bin
    for i in range(len(sqi_bins) - 1):
        bin_mask = (sqi >= sqi_bins[i]) & (sqi < sqi_bins[i + 1])
        
        # Handle last bin to include maximum value
        if i == len(sqi_bins) - 2:
            bin_mask = (sqi >= sqi_bins[i]) & (sqi <= sqi_bins[i + 1])
        
        n_samples = bin_mask.sum()
        sample_counts.append(n_samples)
        
        if n_samples == 0:
            # No samples in this bin, append NaN
            for metric_name in metrics_dict:
                metrics_dict[metric_name].append(np.nan)
        else:
            # Compute metrics for this bin
            bin_preds = predictions[bin_mask]
            bin_targets = targets[bin_mask]
            bin_logits = logits[bin_mask]
            
            bin_metrics = compute_metrics(
                bin_preds, bin_targets, bin_logits, task, num_classes
            )
            
            for metric_name in metrics_dict:
                metrics_dict[metric_name].append(bin_metrics[metric_name])
    
    # Convert to numpy arrays
    for metric_name in metrics_dict:
        metrics_dict[metric_name] = np.array(metrics_dict[metric_name])
    sample_counts = np.array(sample_counts)
    
    return bin_centers, metrics_dict, sample_counts


def compute_quantile_stratified_metrics(data: Dict[str, np.ndarray],
                                        task: str = "multiclass",
                                        num_classes: int = 3) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute metrics for 3 equal-sized SQI bins (1/3, 2/3, 100%).
    
    Args:
        data: Dictionary containing test data
        task: Task type ("binary" or "multiclass")
        num_classes: Number of classes
        
    Returns:
        Tuple of (bin_labels, metrics_dict, sample_counts, sqi_thresholds)
        - bin_labels: Labels for each bin (e.g., ["Low", "Medium", "High"])
        - metrics_dict: Dictionary with metric names as keys and arrays of values per bin
        - sample_counts: Number of samples in each bin
        - sqi_thresholds: SQI thresholds [33.33% quantile, 66.67% quantile, 100%]
    """
    predictions = torch.from_numpy(data['predictions'])
    targets = torch.from_numpy(data['targets'])
    logits = torch.from_numpy(data['logits'])
    sqi = data['sqi']
    
    # Sort indices by SQI to create equal-sized bins
    sorted_indices = np.argsort(sqi)
    n_samples = len(sqi)
    
    # Split into 3 equal parts
    n_per_bin = n_samples // 3
    bin_indices = [
        sorted_indices[:n_per_bin],  # First third
        sorted_indices[n_per_bin:2*n_per_bin],  # Second third
        sorted_indices[2*n_per_bin:]  # Remaining samples (could be slightly more if not divisible by 3)
    ]
    
    # Get SQI thresholds for each bin (use max value in each bin as threshold)
    sqi_min = sqi.min()
    sqi_max = sqi.max()
    thresholds = [sqi_min]
    
    # Calculate actual thresholds from sorted SQI values
    # Use the maximum SQI value in each bin as the threshold
    if len(bin_indices[0]) > 0:
        thresholds.append(np.max(sqi[bin_indices[0]]))
    else:
        thresholds.append(sqi_min)
    
    if len(bin_indices[1]) > 0:
        thresholds.append(np.max(sqi[bin_indices[1]]))
    else:
        thresholds.append(thresholds[-1])
    
    thresholds.append(sqi_max)
    
    # Ensure thresholds are in ascending order and unique
    for i in range(len(thresholds) - 1):
        if thresholds[i] >= thresholds[i + 1]:
            # If they're equal, slightly increase the next threshold
            thresholds[i + 1] = thresholds[i] + 0.01
    
    # Initialize metric arrays
    metrics_dict = {
        'accuracy': [],
        'f1': [],
        'sensitivity': [],
        'ppv': [],
        'specificity': [],
        'auc': []
    }
    sample_counts = []
    bin_labels = []
    
    # Compute metrics for each bin
    for i in range(3):
        bin_mask = np.zeros(n_samples, dtype=bool)
        bin_mask[bin_indices[i]] = True
        
        n_samples_in_bin = bin_mask.sum()
        sample_counts.append(n_samples_in_bin)
        
        # Create bin label
        if i == 0:
            bin_label = f"Low SQI\n[{thresholds[0]:.1f}, {thresholds[1]:.1f}]"
        elif i == 1:
            bin_label = f"Medium SQI\n({thresholds[1]:.1f}, {thresholds[2]:.1f}]"
        else:
            bin_label = f"High SQI\n({thresholds[2]:.1f}, {thresholds[3]:.1f}]"
        bin_labels.append(bin_label)
        
        if n_samples_in_bin == 0:
            # No samples in this bin, append NaN
            for metric_name in metrics_dict:
                metrics_dict[metric_name].append(np.nan)
        else:
            # Compute metrics for this bin
            bin_preds = predictions[bin_mask]
            bin_targets = targets[bin_mask]
            bin_logits = logits[bin_mask]
            
            bin_metrics = compute_metrics(
                bin_preds, bin_targets, bin_logits, task, num_classes
            )
            
            for metric_name in metrics_dict:
                metrics_dict[metric_name].append(bin_metrics[metric_name])
    
    # Convert to numpy arrays
    for metric_name in metrics_dict:
        metrics_dict[metric_name] = np.array(metrics_dict[metric_name])
    sample_counts = np.array(sample_counts)
    
    # Return bin centers (representative SQI values) for plotting
    bin_centers = np.array([
        (thresholds[0] + thresholds[1]) / 2,  # Low bin center
        (thresholds[1] + thresholds[2]) / 2,  # Medium bin center
        (thresholds[2] + thresholds[3]) / 2   # High bin center
    ])
    
    return bin_centers, metrics_dict, sample_counts, np.array(thresholds)


def plot_quantile_metrics_vs_sqi(bin_centers: np.ndarray, metrics_dict: Dict[str, np.ndarray],
                                 sample_counts: np.ndarray, sqi_thresholds: np.ndarray,
                                 save_path: str = None):
    """
    Plot performance metrics vs quantile-based SQI bins (3 equal-sized bins).
    
    Args:
        bin_centers: Representative SQI values for each bin
        metrics_dict: Dictionary of metric arrays
        sample_counts: Number of samples in each bin
        sqi_thresholds: SQI thresholds [min, 33.33%, 66.67%, max]
        save_path: Path to save the figure (optional)
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Performance Metrics vs SQI (Equal-Sized Bins: 1/3, 2/3, 100%)', 
                 fontsize=16, fontweight='bold')
    
    # Define x positions (using bin centers as x-axis)
    x_positions = np.arange(len(bin_centers))
    
    # Plot metrics as grouped bars or lines
    metric_configs = [
        ('accuracy', 'Accuracy'),
        ('auc', 'AUC'),
        ('f1', 'F1 Score'),
        ('sensitivity', 'Sensitivity (Recall)'),
        ('ppv', 'PPV (Precision)'),
        ('specificity', 'Specificity')
    ]
    
    # Define colors for metrics
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_configs)))
    
    # Width for bars
    bar_width = 0.15
    n_metrics = len([m for m, _ in metric_configs if m in metrics_dict])
    
    # Plot each metric as bars (grouped)
    for idx, (metric_key, metric_label) in enumerate(metric_configs):
        if metric_key in metrics_dict:
            values = metrics_dict[metric_key]
            # Offset x positions for grouped bars
            x_offset = (idx - n_metrics / 2) * bar_width + bar_width / 2
            
            # Mask NaN values
            valid_mask = ~np.isnan(values)
            bars = ax.bar(x_positions[valid_mask] + x_offset, values[valid_mask],
                         width=bar_width, label=metric_label, color=colors[idx],
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for i, (x_pos, val) in enumerate(zip(x_positions[valid_mask], values[valid_mask])):
                if val > 0:
                    ax.text(x_pos + x_offset, val, f'{val:.3f}',
                           ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Set x-axis labels
    bin_labels = [
        f"Low SQI\n[{sqi_thresholds[0]:.1f}, {sqi_thresholds[1]:.1f}]",
        f"Medium SQI\n({sqi_thresholds[1]:.1f}, {sqi_thresholds[2]:.1f}]",
        f"High SQI\n({sqi_thresholds[2]:.1f}, {sqi_thresholds[3]:.1f}]"
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('SQI Bins (Equal Sample Sizes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    # Add sample count annotations
    for i, count in enumerate(sample_counts):
        ax.text(x_positions[i], 1.02, f'n={int(count)}',
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved quantile-based plot to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_metrics_vs_sqi(bin_centers: np.ndarray, metrics_dict: Dict[str, np.ndarray],
                        sample_counts: np.ndarray, save_path: str = None):
    """
    Create a single plot showing all metrics and SQI relative frequency vs SQI.
    
    Args:
        bin_centers: Center values of each SQI bin
        metrics_dict: Dictionary of metric arrays
        sample_counts: Number of samples in each bin
        save_path: Path to save the figure (optional)
    """
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Test Performance Metrics and SQI Distribution vs SQI', fontsize=16, fontweight='bold')
    
    # Calculate relative frequency for SQI distribution
    total_samples = sample_counts.sum()
    relative_frequency = sample_counts / total_samples if total_samples > 0 else sample_counts
    
    # Plot metrics on left y-axis
    metric_configs = [
        ('accuracy', 'Accuracy'),
        ('auc', 'AUC'),
        ('f1', 'F1 Score'),
        ('sensitivity', 'Sensitivity (Recall)'),
        ('ppv', 'PPV (Precision)'),
        ('specificity', 'Specificity')
    ]
    
    # Define colors for metrics
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_configs)))
    
    for idx, (metric_key, metric_label) in enumerate(metric_configs):
        if metric_key in metrics_dict:
            values = metrics_dict[metric_key]
            # Mask NaN values for plotting
            valid_mask = ~np.isnan(values)
            
            ax1.plot(bin_centers[valid_mask], values[valid_mask], 
                    marker='o', linewidth=2, markersize=8, 
                    label=metric_label, color=colors[idx], linestyle='-')
    
    ax1.set_xlabel('SQI (Signal Quality Index)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    
    # Plot SQI relative frequency on right y-axis
    ax2 = ax1.twinx()
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 10
    
    # Plot bars for relative frequency
    bars = ax2.bar(bin_centers, relative_frequency, width=bin_width*0.8, 
                   alpha=0.3, color='gray', edgecolor='black', linewidth=1.5,
                   label='SQI Relative Frequency')
    
    ax2.set_ylabel('Relative Frequency', fontsize=12, fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim([0, relative_frequency.max() * 1.2 if relative_frequency.max() > 0 else 1.0])
    
    # Add legend for SQI distribution
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add value labels on bars for relative frequency
    for i, (center, freq) in enumerate(zip(bin_centers, relative_frequency)):
        if freq > 0:
            ax2.text(center, freq, f'{freq:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze test performance stratified by SQI'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the test_sqi_data.npz file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save output figures (default: same directory as data file)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='multiclass',
        choices=['binary', 'multiclass'],
        help='Task type: binary or multiclass'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=3,
        help='Number of classes (default: 3)'
    )
    parser.add_argument(
        '--bin_width',
        type=int,
        default=20,
        help='Width of SQI bins (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading test data from {args.data_path}")
    data = load_test_data(args.data_path)
    print(f"Loaded {len(data['predictions'])} test samples")
    print(f"SQI range: {data['sqi'].min():.2f} - {data['sqi'].max():.2f}")
    
    # Create bins
    sqi_min = int(np.floor(data['sqi'].min() / args.bin_width) * args.bin_width)
    sqi_max = int(np.ceil(data['sqi'].max() / args.bin_width) * args.bin_width)
    sqi_bins = np.arange(sqi_min, sqi_max + args.bin_width, args.bin_width)
    
    # Compute stratified metrics
    print("Computing metrics for each SQI bin...")
    bin_centers, metrics_dict, sample_counts = compute_stratified_metrics(
        data, sqi_bins=sqi_bins, task=args.task, num_classes=args.num_classes
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Summary of Metrics by SQI Bin:")
    print("="*80)
    print(f"{'SQI Range':<15} {'N':<8} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Sens':<10} {'PPV':<10} {'Spec':<10}")
    print("-"*80)
    
    for i in range(len(bin_centers)):
        sqi_range = f"{sqi_bins[i]:.0f}-{sqi_bins[i+1]:.0f}"
        n = int(sample_counts[i])
        acc = metrics_dict['accuracy'][i]
        auc = metrics_dict['auc'][i]
        f1 = metrics_dict['f1'][i]
        sens = metrics_dict['sensitivity'][i]
        ppv = metrics_dict['ppv'][i]
        spec = metrics_dict['specificity'][i]
        
        print(f"{sqi_range:<15} {n:<8} {acc:<10.4f} {auc:<10.4f} {f1:<10.4f} "
              f"{sens:<10.4f} {ppv:<10.4f} {spec:<10.4f}")
    
    # Determine output path
    if args.output_path is None:
        data_dir = os.path.dirname(args.data_path)
        args.output_path = os.path.join(data_dir, 'sqi_performance_analysis.png')
    
    # Create plots
    print("\nGenerating plots...")
    plot_metrics_vs_sqi(bin_centers, metrics_dict, sample_counts, args.output_path)
    
    # Compute quantile-based metrics (3 equal-sized bins)
    print("\nComputing metrics for equal-sized SQI bins (1/3, 2/3, 100%)...")
    quantile_bin_centers, quantile_metrics_dict, quantile_sample_counts, sqi_thresholds = \
        compute_quantile_stratified_metrics(data, task=args.task, num_classes=args.num_classes)
    
    # Print summary for quantile-based bins
    print("\n" + "="*80)
    print("Summary of Metrics by Equal-Sized SQI Bins (1/3, 2/3, 100%):")
    print("="*80)
    print(f"{'SQI Range':<30} {'N':<8} {'Accuracy':<10} {'AUC':<10} {'F1':<10} {'Sens':<10} {'PPV':<10} {'Spec':<10}")
    print("-"*80)
    
    bin_range_labels = [
        f"[{sqi_thresholds[0]:.1f}, {sqi_thresholds[1]:.1f}]",
        f"({sqi_thresholds[1]:.1f}, {sqi_thresholds[2]:.1f}]",
        f"({sqi_thresholds[2]:.1f}, {sqi_thresholds[3]:.1f}]"
    ]
    
    for i in range(len(quantile_bin_centers)):
        sqi_range = bin_range_labels[i]
        n = int(quantile_sample_counts[i])
        acc = quantile_metrics_dict['accuracy'][i]
        auc = quantile_metrics_dict['auc'][i]
        f1 = quantile_metrics_dict['f1'][i]
        sens = quantile_metrics_dict['sensitivity'][i]
        ppv = quantile_metrics_dict['ppv'][i]
        spec = quantile_metrics_dict['specificity'][i]
        
        print(f"{sqi_range:<30} {n:<8} {acc:<10.4f} {auc:<10.4f} {f1:<10.4f} "
              f"{sens:<10.4f} {ppv:<10.4f} {spec:<10.4f}")
    
    # Create quantile-based plot
    quantile_output_path = args.output_path.replace('.png', '_quantile.png') if args.output_path else None
    if quantile_output_path is None:
        data_dir = os.path.dirname(args.data_path)
        quantile_output_path = os.path.join(data_dir, 'sqi_performance_analysis_quantile.png')
    
    print("\nGenerating quantile-based plot...")
    plot_quantile_metrics_vs_sqi(quantile_bin_centers, quantile_metrics_dict, 
                                 quantile_sample_counts, sqi_thresholds, 
                                 quantile_output_path)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

