"""
Cross-validation script to test model generalization across multiple datasets.

This script:
1. Finds all datasets under ./PPG_data/splitted_data/
2. For each dataset, runs training, validation, and test
3. Collects metrics from each run
4. Computes mean and standard deviation across all datasets
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import pytorch_lightning as pl
import torch
import shutil
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from utils import create_train_data_loader, create_test_data_loader, create_balanced_train_data_loader
from models.model_adapt import EctopicsClassifier
from models.loss_plot_callback import LossPlotCallback
from data.augmentations import set_augmentations_seed


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
            train_x = item / "ppg_train_normalized.npy"
            train_y = item / "labels_train.npy"
            if train_x.exists() and train_y.exists():
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
            # Replace {dataset} with actual dataset name
            return path_str.format(dataset=dataset_name)
        return path_str
    
    # Update train paths
    if "data" in cfg_dict and "path" in cfg_dict["data"] and "train" in cfg_dict["data"]["path"]:
        for key in cfg_dict["data"]["path"]["train"]:
            if isinstance(cfg_dict["data"]["path"]["train"][key], str):
                cfg_dict["data"]["path"]["train"][key] = update_path(cfg_dict["data"]["path"]["train"][key])
    
    # Update valid paths
    if "data" in cfg_dict and "path" in cfg_dict["data"] and "valid" in cfg_dict["data"]["path"]:
        for key in cfg_dict["data"]["path"]["valid"]:
            if isinstance(cfg_dict["data"]["path"]["valid"][key], str):
                cfg_dict["data"]["path"]["valid"][key] = update_path(cfg_dict["data"]["path"]["valid"][key])
    
    # Update test paths
    if "data" in cfg_dict and "path" in cfg_dict["data"] and "test" in cfg_dict["data"]["path"]:
        for key in cfg_dict["data"]["path"]["test"]:
            if isinstance(cfg_dict["data"]["path"]["test"][key], str):
                cfg_dict["data"]["path"]["test"][key] = update_path(cfg_dict["data"]["path"]["test"][key])
    
    return OmegaConf.create(cfg_dict)


def train_and_test_model(cfg: DictConfig, dataset_name: str, run_idx: int, total_datasets: int) -> dict:
    """Train and test model for a single dataset, return metrics."""
    print(f"\n{'='*80}")
    print(f"Dataset {run_idx + 1}/{total_datasets}: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Set seed for reproducibility
    seed = 1024
    pl.seed_everything(seed, workers=True)
    set_augmentations_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Update config paths
    cfg = update_config_paths(cfg, dataset_name)
    
    # Load data
    print("Loading data...", flush=True)
    train_loader, valid_loader = create_balanced_train_data_loader(cfg.data)
    test_loader = create_test_data_loader(cfg.data)
    print("Done!", flush=True)
    
    # Create model
    total_training_steps = len(train_loader) * cfg.trainer.parameters.max_epochs
    model = EctopicsClassifier(**cfg.model, total_training_steps=total_training_steps, training_config=cfg)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.trainer.callbacks.early_stop)
    loss_plot_callback = LossPlotCallback()
    callbacks = [checkpoint_callback, early_stop_callback, loss_plot_callback]
    
    # Setup loggers with unique version for each dataset
    logger = TensorBoardLogger(
        save_dir=cfg.trainer.callbacks.logger.save_dir,
        name=cfg.trainer.callbacks.logger.name,
        version=f"crossval_{dataset_name}",
    )
    csv_logger = CSVLogger(
        save_dir=cfg.trainer.callbacks.logger.save_dir,
        name=cfg.trainer.callbacks.logger.name,
        version=f"crossval_{dataset_name}",
    )
    
    trainer = pl.Trainer(**cfg.trainer.parameters, callbacks=callbacks, logger=[logger, csv_logger])
    
    # Train
    if cfg.experiment.train:
        print(f"\nTraining on dataset: {dataset_name}...", flush=True)
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
    
    # Test
    metrics = {}
    if cfg.experiment.test:
        print(f"\nTesting on dataset: {dataset_name}...", flush=True)
        test_ckpt = "best" if cfg.trainer.callbacks.model_checkpoint.save_top_k and cfg.trainer.callbacks.model_checkpoint.monitor else None
        test_results = trainer.test(model=model, dataloaders=test_loader, ckpt_path=test_ckpt)
        
        # Extract metrics from test results
        if test_results and len(test_results) > 0:
            metrics = test_results[0]
    
    # Also try to extract from CSV if available
    csv_path = os.path.join(logger.log_dir, "metrics.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Get the last row (test metrics)
            test_rows = df[df['test_loss'].notna()]
            if len(test_rows) > 0:
                last_test_row = test_rows.iloc[-1]
                # Update metrics with CSV values (they might be more complete)
                for col in last_test_row.index:
                    if col.startswith('test_') and pd.notna(last_test_row[col]):
                        metrics[col] = float(last_test_row[col])
        except Exception as e:
            print(f"Warning: Could not read metrics from CSV: {e}", flush=True)
    
    return metrics


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
                    if isinstance(val, (torch.Tensor, np.ndarray)):
                        val = float(val.item() if hasattr(val, 'item') else val)
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
    print("CROSS-VALIDATION ACROSS MULTIPLE DATASETS")
    print("="*80)
    
    # Load base config to get base_path
    config_path = "config"
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name="train")
    
    # Get base_path from config
    base_path = cfg.data.get("base_path", "./PPG_data/splitted_data/")
    if not base_path.endswith("/"):
        base_path = base_path + "/"
    
    # Find all datasets
    datasets = find_datasets(base_path)
    
    if not datasets:
        print(f"Error: No datasets found under {base_path}")
        print("Expected directories containing ppg_train_normalized.npy and labels_train.npy")
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
            metrics = train_and_test_model(cfg, dataset_name, idx, len(datasets))
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
    individual_path = os.path.join(results_dir, "individual_metrics.csv")
    individual_df.to_csv(individual_path)
    print(f"\nSaved individual metrics to: {individual_path}")
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(results_dir, "mean_std_metrics.csv")
    stats_df.to_csv(stats_path)
    print(f"Saved mean/std statistics to: {stats_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: MEAN ± STD")
    print("="*80)
    
    # Group metrics by type
    metric_groups = {
        "Loss": [k for k in stats.keys() if k.endswith("_mean") and "loss" in k.lower()],
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
    print("Cross-validation completed!")
    print(f"Results saved to: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

