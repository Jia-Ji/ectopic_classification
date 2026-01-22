"""
Script to visualize embeddings using t-SNE.
Shows embeddings before and after attention layer, colored by true class labels.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from sklearn.manifold import TSNE
import seaborn as sns

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')

# Color palette for classes
CLASS_COLORS = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
CLASS_NAMES = ['Normal', 'Ectopic', 'VT']


def load_embeddings(embeddings_path: str) -> dict:
    """
    Load embeddings from saved .npz file.
    
    Args:
        embeddings_path: Path to the .npz file containing embeddings
        
    Returns:
        Dictionary with keys: 'z_backbone', 'z_att', 'labels', 'sqi'
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
    
    data = np.load(embeddings_path)
    return {
        'z_backbone': data['z_backbone'],
        'z_att': data['z_att'],
        'labels': data['labels'],
        'sqi': data['sqi']
    }


def apply_tsne(embeddings: np.ndarray, n_components: int = 2, 
               perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction to embeddings.
    
    Args:
        embeddings: Embeddings array of shape (N, C)
        n_components: Number of dimensions for t-SNE output (default: 2)
        perplexity: Perplexity parameter for t-SNE (default: 30.0)
        random_state: Random state for reproducibility
        
    Returns:
        t-SNE transformed embeddings of shape (N, n_components)
    """
    print(f"Applying t-SNE to embeddings of shape {embeddings.shape}...")
    print(f"  Perplexity: {perplexity}, Random state: {random_state}")
    
    # Adjust perplexity if needed (must be less than n_samples)
    n_samples = embeddings.shape[0]
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
        print(f"  Adjusted perplexity to {perplexity} (n_samples={n_samples})")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=1
    )
    
    tsne_result = tsne.fit_transform(embeddings)
    print(f"t-SNE completed. Output shape: {tsne_result.shape}")
    
    return tsne_result


def plot_embeddings(tsne_result: np.ndarray, labels: np.ndarray, 
                   title: str, save_path: str = None, figsize: tuple = (10, 8)):
    """
    Plot t-SNE embeddings colored by true class labels.
    
    Args:
        tsne_result: t-SNE transformed embeddings of shape (N, 2)
        labels: True class labels of shape (N,)
        title: Title for the plot
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique labels and their counts
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Plot each class with different color
    for i, class_label in enumerate(unique_labels):
        mask = labels == class_label
        class_name = CLASS_NAMES[class_label] if class_label < len(CLASS_NAMES) else f'Class {class_label}'
        color = CLASS_COLORS[class_label] if class_label < len(CLASS_COLORS) else plt.cm.tab10(i)
        
        ax.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[color],
            label=f'{class_name} (n={np.sum(mask)})',
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_side_by_side_plots(tsne_backbone: np.ndarray, tsne_att: np.ndarray,
                              labels: np.ndarray, save_path: str = None,
                              figsize: tuple = (16, 7)):
    """
    Create side-by-side plots comparing embeddings before and after attention.
    
    Args:
        tsne_backbone: t-SNE result for embeddings before attention
        tsne_att: t-SNE result for embeddings after attention
        labels: True class labels
        save_path: Path to save the figure
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Plot before attention
    ax1 = axes[0]
    for i, class_label in enumerate(unique_labels):
        mask = labels == class_label
        class_name = CLASS_NAMES[class_label] if class_label < len(CLASS_NAMES) else f'Class {class_label}'
        color = CLASS_COLORS[class_label] if class_label < len(CLASS_COLORS) else plt.cm.tab10(i)
        
        ax1.scatter(
            tsne_backbone[mask, 0],
            tsne_backbone[mask, 1],
            c=[color],
            label=f'{class_name} (n={np.sum(mask)})',
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('Embeddings Before Attention', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot after attention
    ax2 = axes[1]
    for i, class_label in enumerate(unique_labels):
        mask = labels == class_label
        class_name = CLASS_NAMES[class_label] if class_label < len(CLASS_NAMES) else f'Class {class_label}'
        color = CLASS_COLORS[class_label] if class_label < len(CLASS_COLORS) else plt.cm.tab10(i)
        
        ax2.scatter(
            tsne_att[mask, 0],
            tsne_att[mask, 1],
            c=[color],
            label=f'{class_name} (n={np.sum(mask)})',
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('Embeddings After Attention', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('t-SNE Visualization of Embeddings', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved side-by-side plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize embeddings using t-SNE, colored by true class labels'
    )
    parser.add_argument(
        '--embeddings_path',
        type=str,
        required=True,
        help='Path to the test_embeddings.npz file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output plots (default: same as embeddings file)'
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='Perplexity parameter for t-SNE (default: 30.0)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random state for t-SNE reproducibility (default: 42)'
    )
    parser.add_argument(
        '--side_by_side',
        action='store_true',
        help='Create side-by-side comparison plot instead of separate plots'
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}")
    data = load_embeddings(args.embeddings_path)
    
    z_backbone = data['z_backbone']
    z_att = data['z_att']
    labels = data['labels']
    sqi = data['sqi']
    
    print(f"\nLoaded embeddings:")
    print(f"  z_backbone shape: {z_backbone.shape}")
    print(f"  z_att shape: {z_att.shape}")
    print(f"  labels shape: {labels.shape}")
    print(f"  Number of samples: {len(labels)}")
    print(f"  Class distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        class_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f'Class {int(label)}'
        print(f"    {class_name}: {count}")
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.embeddings_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Apply t-SNE to both embeddings
    print("\n" + "="*80)
    print("Applying t-SNE to embeddings before attention...")
    print("="*80)
    tsne_backbone = apply_tsne(
        z_backbone,
        perplexity=args.perplexity,
        random_state=args.random_state
    )
    
    print("\n" + "="*80)
    print("Applying t-SNE to embeddings after attention...")
    print("="*80)
    tsne_att = apply_tsne(
        z_att,
        perplexity=args.perplexity,
        random_state=args.random_state
    )
    
    # Create visualizations
    if args.side_by_side:
        # Side-by-side comparison
        output_path = os.path.join(args.output_dir, 'embeddings_tsne_comparison.png')
        print(f"\nCreating side-by-side comparison plot...")
        create_side_by_side_plots(tsne_backbone, tsne_att, labels, output_path)
    else:
        # Separate plots
        backbone_path = os.path.join(args.output_dir, 'embeddings_tsne_before_attention.png')
        att_path = os.path.join(args.output_dir, 'embeddings_tsne_after_attention.png')
        
        print(f"\nCreating plot for embeddings before attention...")
        plot_embeddings(
            tsne_backbone,
            labels,
            'Embeddings Before Attention (t-SNE)',
            backbone_path
        )
        
        print(f"\nCreating plot for embeddings after attention...")
        plot_embeddings(
            tsne_att,
            labels,
            'Embeddings After Attention (t-SNE)',
            att_path
        )
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()
