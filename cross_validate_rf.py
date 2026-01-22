"""
Cross-validation script for Random Forest model across multiple datasets.

This script:
1. Finds all datasets under the configured base_path
2. For each dataset, runs training and test
3. Collects metrics from each run
4. Computes mean and standard deviation across all datasets
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize

from utils import load_train_biomarkers_and_labels, load_test_biomarkers_and_labels
from models.random_forest import RandomForest


def find_datasets(base_path: str) -> list:
    """Find all dataset directories under base_path."""
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    # Find all directories that contain the required files
    datasets = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if it has the required training files
            hr_train = item / "hr_train.npy"
            labels_train = item / "labels_train.npy"
            if hr_train.exists() and labels_train.exists():
                datasets.append(item.name)
    
    return sorted(datasets)


def update_config_paths(cfg: DictConfig, dataset_name: str) -> DictConfig:
    """Update config paths with the dataset name."""
    # Get base_path from config (resolve interpolation first)
    base_path = OmegaConf.select(cfg, "data.base_path")
    if base_path is None:
        base_path = "./PPG_data/splitted_data/"
    if not base_path.endswith("/"):
        base_path = base_path + "/"
    
    # Create a mutable copy of the config
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Update paths - replace {dataset} placeholder with actual dataset name
    def update_path(path_str: str) -> str:
        if path_str and "{dataset}" in path_str:
            return path_str.format(dataset=dataset_name)
        return path_str
    
    # Update train paths
    if "data" in cfg_dict and "path" in cfg_dict["data"] and "train" in cfg_dict["data"]["path"]:
        for key in cfg_dict["data"]["path"]["train"]:
            if isinstance(cfg_dict["data"]["path"]["train"][key], str):
                cfg_dict["data"]["path"]["train"][key] = update_path(cfg_dict["data"]["path"]["train"][key])
    
    # Update test paths
    if "data" in cfg_dict and "path" in cfg_dict["data"] and "test" in cfg_dict["data"]["path"]:
        for key in cfg_dict["data"]["path"]["test"]:
            if isinstance(cfg_dict["data"]["path"]["test"][key], str):
                cfg_dict["data"]["path"]["test"][key] = update_path(cfg_dict["data"]["path"]["test"][key])
    
    return OmegaConf.create(cfg_dict)


def train_and_test_rf(cfg: DictConfig, dataset_name: str, run_idx: int, total_datasets: int) -> dict:
    """Train and test Random Forest model for a single dataset, return metrics."""
    print(f"\n{'='*80}")
    print(f"Dataset {run_idx + 1}/{total_datasets}: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Update config paths
    cfg = update_config_paths(cfg, dataset_name)
    
    # Extract paths from config
    train_paths = cfg.data.path.train
    test_paths = cfg.data.path.test
    
    # Load training features and labels
    print("Loading training data...", flush=True)
    train_features, train_labels = load_train_biomarkers_and_labels(
        hr_path=train_paths.hr_path,
        hrv_path=train_paths.hrv_path,
        y_path=train_paths.y_path
    )
    print(f"Training data loaded: {len(train_features)} samples")
    
    # Load test features and labels
    print("Loading test data...", flush=True)
    test_features, test_labels = load_test_biomarkers_and_labels(
        hr_path=test_paths.hr_path,
        hrv_path=test_paths.hrv_path,
        y_path=test_paths.y_path
    )
    print(f"Test data loaded: {len(test_features)} samples")
    
    # Initialize Random Forest model
    print("\nInitializing Random Forest model...", flush=True)
    model = RandomForest(
        n_feats_to_select=cfg.model.n_feats_to_select,
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        min_samples_split=cfg.model.min_samples_split,
        min_samples_leaf=cfg.model.min_samples_leaf,
        class_weight=cfg.model.class_weight,
        random_state=cfg.model.random_state,
        get_summary=cfg.model.get_summary,
        confusion_matrix=cfg.model.confusion_matrix,
        classification_report=cfg.model.classification_report,
        aus_score=cfg.model.aus_score,
        plot_roc=False  # Disable plotting for cross-validation
    )
    
    # Train the model
    print(f"\nTraining Random Forest on dataset: {dataset_name}...", flush=True)
    model.train(train_features, train_labels)
    
    # Test the model
    print(f"\nTesting Random Forest on dataset: {dataset_name}...", flush=True)
    metrics = model.test(test_features, test_labels)
    
    return metrics if metrics else {}


def compute_statistics(all_metrics: list) -> dict:
    """Compute mean and std across all runs."""
    if not all_metrics:
        return {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    
    stats = {}
    for key in all_keys:
        values = []
        for metrics in all_metrics:
            if key in metrics:
                val = metrics[key]
                # Convert to float if possible
                try:
                    if isinstance(val, (np.ndarray, list)):
                        val = float(val[0] if len(val) > 0 else 0)
                    else:
                        val = float(val)
                    values.append(val)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values
        
        if values:
            stats[f"{key}_mean"] = np.mean(values)
            stats[f"{key}_std"] = np.std(values)
            stats[f"{key}_values"] = values  # Keep individual values for reference
    
    return stats


def main():
    """Main function to run cross-validation across all datasets."""
    print("="*80)
    print("RANDOM FOREST CROSS-VALIDATION ACROSS MULTIPLE DATASETS")
    print("="*80)
    
    # Load base config to get base_path
    config_path = "config"
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name="train_rf_config")
    
    # Get base_path from config
    base_path = cfg.data.get("base_path", "./PPG_data/splitted_data/")
    if not base_path.endswith("/"):
        base_path = base_path + "/"
    
    # Find all datasets
    datasets = find_datasets(base_path)
    
    if not datasets:
        print(f"Error: No datasets found under {base_path}")
        print("Expected directories containing hr_train.npy and labels_train.npy")
        sys.exit(1)
    
    print(f"\nFound {len(datasets)} datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    print(f"\nUsing base path: {base_path}")
    
    # Run training and testing for each dataset
    all_metrics = []
    dataset_names = []
    
    for idx, dataset_name in enumerate(datasets):
        try:
            metrics = train_and_test_rf(cfg, dataset_name, idx, len(datasets))
            if metrics:
                all_metrics.append(metrics)
                dataset_names.append(dataset_name)
                print(f"\n✓ Completed dataset {dataset_name}")
                print(f"  Test metrics: {list(metrics.keys())[:5]}...")  # Show first 5 keys
            else:
                print(f"\n⚠ Warning: No metrics collected for dataset {dataset_name}")
        except Exception as e:
            print(f"\n✗ Error processing dataset {dataset_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    # Compute statistics
    if not all_metrics:
        print("\nError: No metrics collected from any dataset!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("COMPUTING STATISTICS")
    print("="*80)
    
    stats = compute_statistics(all_metrics)
    
    # Save results
    results_dir = "logs/cross_validation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual metrics
    individual_df = pd.DataFrame(all_metrics, index=dataset_names)
    individual_path = os.path.join(results_dir, "rf_individual_metrics.csv")
    individual_df.to_csv(individual_path)
    print(f"\nSaved individual metrics to: {individual_path}")
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(results_dir, "rf_mean_std_metrics.csv")
    stats_df.to_csv(stats_path)
    print(f"Saved mean/std statistics to: {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: MEAN ± STD")
    print("="*80)
    
    # Group metrics by type
    metric_groups = {
        "Accuracy": [k for k in stats.keys() if k.endswith("_mean") and "accuracy" in k.lower()],
        "F1": [k for k in stats.keys() if k.endswith("_mean") and "f1" in k.lower()],
        "Sensitivity": [k for k in stats.keys() if k.endswith("_mean") and "sensitivity" in k.lower()],
        "Specificity": [k for k in stats.keys() if k.endswith("_mean") and "specificity" in k.lower()],
        "PPV": [k for k in stats.keys() if k.endswith("_mean") and "ppv" in k.lower()],
        "AUC": [k for k in stats.keys() if k.endswith("_mean") and "auc" in k.lower()],
    }
    
    for group_name, metric_keys in metric_groups.items():
        if metric_keys:
            print(f"\n{group_name}:")
            for key in sorted(metric_keys):
                mean_key = key
                std_key = key.replace("_mean", "_std")
                if std_key in stats:
                    mean_val = stats[mean_key]
                    std_val = stats[std_key]
                    metric_name = key.replace("_mean", "").replace("test_", "")
                    print(f"  {metric_name:40s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n" + "="*80)
    print("Random Forest Cross-validation completed!")
    print(f"Results saved to: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

