"""
Main entry point for the PPG Ectopics Classification Pipeline.

This module provides eight main pipelines:
1. build_dataset - Build dataset with biomarkers extraction
2. preprocess_signals - Preprocess PPG/ECG signals
3. split_data - Split data into train/val/test sets
4. domain_normalize - Normalize PPG data by domain (Cathlab vs Theatre)
5. train_model - Train the classification model
6. train_regression - Train the regression model
7. train_svm - Train the SVM model
8. train_random_forest - Train the Random Forest model

Usage:
    python main.py build_dataset    # Run dataset building pipeline
    python main.py preprocess       # Run preprocessing pipeline
    python main.py split            # Run data splitting pipeline
    python main.py domain_normalize # Run domain normalization pipeline
    python main.py train            # Run training pipeline
    python main.py train_regression # Run regression training pipeline
    python main.py train_svm        # Run SVM training pipeline
    python main.py train_random_forest # Run Random Forest training pipeline
    python main.py evaluate         # Run evaluation pipeline
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import sys
import os

# Pipeline imports
from data_pipeline.dataset_builder import CombinedDataBuilder
from data_pipeline.signal_preprocessor import Preprocessor
from data_pipeline.data_splitter import DataSplitter
from data_pipeline.domain_normalizer import DomainNormalizer

# Training imports
import hydra.utils
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import shutil
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from utils import create_train_data_loader, create_test_data_loader, create_balanced_train_data_loader, load_train_biomarkers_and_labels, load_test_biomarkers_and_labels
from models.model_adapt import EctopicsClassifier
from models.regression import Regression
from models.svm import SVM
from models.random_forest import RandomForest
from models.loss_plot_callback import LossPlotCallback
from data.augmentations import set_augmentations_seed


def build_dataset_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 1: Build dataset with biomarkers extraction.
    
    This pipeline:
    - Extracts biomarkers (SQI, HR, fiducials) from PPG/ECG signals
    - Filters data by multiple SQI thresholds
    - Plots noisy examples
    """
    print("=" * 80)
    print("PIPELINE 1: DATASET BUILDING WITH BIOMARKERS")
    print("=" * 80)
    
    builder = CombinedDataBuilder(**cfg.extractor)
    combined_df = builder.run_pipeline(cfg)
    
    print("\n✓ Dataset building pipeline completed!")


def preprocess_signals_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 2: Preprocess PPG/ECG signals.
    
    This pipeline:
    - Applies bandpass filtering
    - Clips outliers using IQR method
    - Normalizes signals by patient ID
    """
    print("=" * 80)
    print("PIPELINE 2: SIGNAL PREPROCESSING")
    print("=" * 80)
    
    preprocessor = Preprocessor(cfg)
    preprocessor.run()
    
    print("\n✓ Signal preprocessing pipeline completed!")


def split_data_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 3: Split data into train/validation/test sets.
    
    This pipeline:
    - Filters and encodes labels (merges PAC/PVC into ECT)
    - Splits data by patient ID to avoid data leakage
    - Saves splits as numpy arrays
    - Generates distribution reports
    """
    print("=" * 80)
    print("PIPELINE 3: DATA SPLITTING")
    print("=" * 80)
    
    splitter = DataSplitter(cfg)
    splitter.run()
    
    print("\n✓ Data splitting pipeline completed!")


def domain_normalize_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 4: Domain-based normalization for PPG signals.
    
    This pipeline:
    - Loads ID0 and PPG data from splitted data
    - Identifies domain based on ID0 prefix (ID = Cathlab, Theatre = Cardiac Theatre)
    - Computes domain statistics (mean, std) from training set only
    - Normalizes all splits (train, val, test) by domain statistics
    - Saves normalized data and domain statistics
    """
    print("=" * 80)
    print("PIPELINE 4: DOMAIN-BASED NORMALIZATION")
    print("=" * 80)
    
    normalizer = DomainNormalizer(cfg)
    normalizer.run()
    
    print("\n✓ Domain normalization pipeline completed!")


def train_model_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 5: Train and evaluate the classification model.
    
    This pipeline:
    - Loads train/val/test data
    - Trains the model with balanced augmentation
    - Evaluates on test set
    - Saves logs and checkpoints
    """
    print("=" * 80)
    print("PIPELINE 4: MODEL TRAINING")
    print("=" * 80)
    
    # Set seed FIRST before anything else
    seed = 1024
    pl.seed_everything(seed, workers=True)
    
    # Set augmentation RNG seed for reproducibility
    set_augmentations_seed(seed)
    
    # Additional deterministic settings for full reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Data loading ...", flush=True)
    train_loader, valid_loader = create_balanced_train_data_loader(cfg.data)
    test_loader = create_test_data_loader(cfg.data)
    print("Done!", flush=True)

    total_training_steps = len(train_loader) * cfg.trainer.parameters.max_epochs

    model = EctopicsClassifier(**cfg.model, total_training_steps=total_training_steps, training_config=cfg)

    checkpoint_callback = ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.trainer.callbacks.early_stop)
    loss_plot_callback = LossPlotCallback()
    callbacks = [checkpoint_callback, early_stop_callback, loss_plot_callback]

    logger = TensorBoardLogger(**cfg.trainer.callbacks.logger)
    # Align CSV logger with TensorBoard logger directory and version
    csv_logger = CSVLogger(
        save_dir=cfg.trainer.callbacks.logger.save_dir,
        name=cfg.trainer.callbacks.logger.name,
        version=logger.version,
    )

    trainer = pl.Trainer(**cfg.trainer.parameters, callbacks=callbacks, logger=[logger, csv_logger])

    ckpt_path = None
    if cfg.experiment.resume_ckpt:
        ckpt_path = cfg.experiment.ckpt_path

    if cfg.experiment.train:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=ckpt_path,
        )

    if cfg.experiment.test:
        # Use best checkpoint if available; otherwise test current model
        test_ckpt = "best" if cfg.trainer.callbacks.model_checkpoint.save_top_k and cfg.trainer.callbacks.model_checkpoint.monitor else None
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=test_ckpt)
        
    # Save main_log.txt to logs directory after testing
    original_cwd = hydra.utils.get_original_cwd()
    main_log_path = os.path.join(original_cwd, "main_log.txt")
    
    if os.path.exists(main_log_path):
        # Get the logger version directory
        version = logger.version if logger.version is not None else ""
        if version:
            log_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
        else:
            log_dir = os.path.join(logger.save_dir, logger.name)
        
        os.makedirs(log_dir, exist_ok=True)
        dest_path = os.path.join(log_dir, "main_log.txt")
        shutil.copy2(main_log_path, dest_path)
        print(f"Saved main_log.txt to {dest_path}", flush=True)
    else:
        print(f"Warning: main_log.txt not found in current or original directory. Skipping copy to logs directory.", flush=True)
    
    print("\n✓ Model training pipeline completed!")

def finetune_model_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline: Finetune a pretrained model on a new dataset.
    
    This pipeline:
    - Loads a pretrained checkpoint
    - Optionally freezes feature extractor (only trains classifier)
    - Optionally reinitializes classifier weights
    - Trains on the new dataset
    - Evaluates on test set
    - Saves logs and checkpoints
    """
    print("=" * 80)
    print("FINETUNING PIPELINE")
    print("=" * 80)
    
    # Set seed for reproducibility
    seed = 1024
    pl.seed_everything(seed, workers=True)
    set_augmentations_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Validate finetuning config
    finetune_cfg = cfg.get('finetune', {})
    if not finetune_cfg.get('enable', False):
        print("Warning: Finetuning is not enabled in config. Set finetune.enable=true")
        return
    
    pretrained_ckpt_path = finetune_cfg.get('pretrained_ckpt_path', None)
    if pretrained_ckpt_path is None or not os.path.exists(pretrained_ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt_path}. "
                                "Please set finetune.pretrained_ckpt_path in config.")
    
    freeze_feature_extractor = finetune_cfg.get('freeze_feature_extractor', True)
    reinit_classifier = finetune_cfg.get('reinit_classifier', False)
    feature_extractor_lr_scale = finetune_cfg.get('feature_extractor_lr_scale', 0.1)
    
    print(f"Pretrained checkpoint: {pretrained_ckpt_path}", flush=True)
    print(f"Freeze feature extractor: {freeze_feature_extractor}", flush=True)
    print(f"Reinitialize classifier: {reinit_classifier}", flush=True)
    print(f"Feature extractor LR scale: {feature_extractor_lr_scale}", flush=True)

    # Load data
    print("\nData loading ...", flush=True)
    train_loader, valid_loader = create_balanced_train_data_loader(cfg.data)
    test_loader = create_test_data_loader(cfg.data)
    print("Done!", flush=True)

    total_training_steps = len(train_loader) * cfg.trainer.parameters.max_epochs

    # Load pretrained checkpoint to extract model state
    print(f"\nLoading pretrained model from: {pretrained_ckpt_path}", flush=True)
    checkpoint = torch.load(pretrained_ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract model config from saved hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    model_hparams = hparams.get('model', {})
    
    # Use the model config from the pretrained checkpoint
    model_config = model_hparams.get('config', None)
    if model_config is None:
        print("Warning: Could not find model config in checkpoint. Using config from finetune.yaml", flush=True)
        model_config = cfg.model.config
    else:
        from omegaconf import OmegaConf
        model_config = OmegaConf.create(model_config)
    
    # Create model with finetuning parameters
    # Note: Override some parameters from current config (lr, num_classes, etc.)
    model = EctopicsClassifier(
        task=cfg.model.task,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        loss_name=cfg.model.loss_name,
        use_lr_scheduler=cfg.model.use_lr_scheduler,
        lr_warmup_ratio=cfg.model.lr_warmup_ratio,
        device=cfg.model.device,
        total_training_steps=total_training_steps,
        config=model_config,  # Use pretrained model architecture
        training_config=cfg,
        class_weights=cfg.model.get('class_weights', None),
        ectopic_threshold=cfg.model.get('ectopic_threshold', None),
        # Finetuning parameters
        freeze_feature_extractor=freeze_feature_extractor,
        reinit_classifier=reinit_classifier,
        feature_extractor_lr_scale=feature_extractor_lr_scale,
    )
    
    # Load pretrained weights
    state_dict = checkpoint['state_dict']
    
    # If reinitializing classifier, only load feature extractor weights
    if reinit_classifier:
        print("Loading only feature extractor weights (classifier will be reinitialized)...", flush=True)
        # Filter out classifier weights
        state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        model.load_state_dict(state_dict, strict=False)
    else:
        # Load all weights
        print("Loading all pretrained weights...", flush=True)
        model.load_state_dict(state_dict, strict=True)
    
    print("Pretrained weights loaded successfully!", flush=True)

    # Setup callbacks and loggers
    checkpoint_callback = ModelCheckpoint(**cfg.trainer.callbacks.model_checkpoint)
    early_stop_callback = EarlyStopping(**cfg.trainer.callbacks.early_stop)
    loss_plot_callback = LossPlotCallback()
    callbacks = [checkpoint_callback, early_stop_callback, loss_plot_callback]

    logger = TensorBoardLogger(**cfg.trainer.callbacks.logger)
    csv_logger = CSVLogger(
        save_dir=cfg.trainer.callbacks.logger.save_dir,
        name=cfg.trainer.callbacks.logger.name,
        version=logger.version,
    )

    trainer = pl.Trainer(**cfg.trainer.parameters, callbacks=callbacks, logger=[logger, csv_logger])

    # Train (finetune)
    if cfg.experiment.train:
        print("\nStarting finetuning...", flush=True)
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

    # Test
    if cfg.experiment.test:
        test_ckpt = "best" if cfg.trainer.callbacks.model_checkpoint.save_top_k and cfg.trainer.callbacks.model_checkpoint.monitor else None
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=test_ckpt)
        
    # Save logs
    original_cwd = hydra.utils.get_original_cwd()
    main_log_path = os.path.join(original_cwd, "main_log.txt")
    
    if os.path.exists(main_log_path):
        version = logger.version if logger.version is not None else ""
        if version:
            log_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
        else:
            log_dir = os.path.join(logger.save_dir, logger.name)
        
        os.makedirs(log_dir, exist_ok=True)
        dest_path = os.path.join(log_dir, "main_log.txt")
        shutil.copy2(main_log_path, dest_path)
        print(f"Saved main_log.txt to {dest_path}", flush=True)
    
    # Save finetuning info
    version = logger.version if logger.version is not None else ""
    if version:
        log_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
    else:
        log_dir = os.path.join(logger.save_dir, logger.name)
    
    finetune_info_path = os.path.join(log_dir, "finetune_info.txt")
    with open(finetune_info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINETUNING INFO\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Pretrained checkpoint: {pretrained_ckpt_path}\n")
        f.write(f"Freeze feature extractor: {freeze_feature_extractor}\n")
        f.write(f"Reinitialize classifier: {reinit_classifier}\n")
        f.write(f"Feature extractor LR scale: {feature_extractor_lr_scale}\n")
        f.write(f"Learning rate: {cfg.model.lr}\n")
        f.write(f"Max epochs: {cfg.trainer.parameters.max_epochs}\n")
    print(f"Saved finetune info to {finetune_info_path}", flush=True)
    
    print("\n✓ Finetuning pipeline completed!")

def evaluate_model_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 6: Evaluate a trained classification model on test data.
    
    This pipeline:
    - Loads test data from the path specified in evaluate.yaml
    - Loads trained model from checkpoint
    - Evaluates on test set
    - Saves logs and evaluation metrics
    """
    print("=" * 80)
    print("PIPELINE 6: MODEL EVALUATION")
    print("=" * 80)
    
    # Set seed for reproducibility
    seed = 1024
    pl.seed_everything(seed, workers=True)
    
    # Set augmentation RNG seed for reproducibility
    set_augmentations_seed(seed)

    print("Loading test data ...", flush=True)
    test_loader = create_test_data_loader(cfg.data)
    print(f"Test data loaded: {len(test_loader)} batches", flush=True)
    
    # Verify checkpoint path exists
    ckpt_path = cfg.experiment.ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    print(f"Loading model from checkpoint: {ckpt_path}", flush=True)
    
    # Load checkpoint to extract config (needed for model initialization)
    checkpoint = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract model config from saved hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    model_hparams = hparams.get('model', {})
    
    model_config = model_hparams.get('config', None)
    if model_config is None:
        raise ValueError("Could not find model config in checkpoint. Make sure the checkpoint was saved with training config.")
    
    # Extract all model parameters that need to be passed explicitly
    from omegaconf import OmegaConf
    model_kwargs = {
        'config': OmegaConf.create(model_config),
        'task': model_hparams.get('task', 'multiclass'),
        'num_classes': model_hparams.get('num_classes', 3),
        'lr': model_hparams.get('lr', 0.0005),
        'weight_decay': model_hparams.get('weight_decay', 0.00003),
        'loss_name': model_hparams.get('loss_name', 'cross_entropy'),
        'use_lr_scheduler': model_hparams.get('use_lr_scheduler', False),
        'lr_warmup_ratio': model_hparams.get('lr_warmup_ratio', 0.01),
        'class_weights': model_hparams.get('class_weights', None),
        'ectopic_threshold': model_hparams.get('ectopic_threshold', None),
    }
    
    print(f"Model params: task={model_kwargs['task']}, num_classes={model_kwargs['num_classes']}", flush=True)
    
    # Load model from checkpoint with explicit parameters
    model = EctopicsClassifier.load_from_checkpoint(
        ckpt_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        **model_kwargs
    )
    print("Model loaded successfully!", flush=True)
    
    # Setup loggers for evaluation
    logger = TensorBoardLogger(**cfg.trainer.callbacks.logger)
    csv_logger = CSVLogger(
        save_dir=cfg.trainer.callbacks.logger.save_dir,
        name=cfg.trainer.callbacks.logger.name,
        version=logger.version,
    )
    
    # Create trainer for evaluation (no training callbacks needed)
    trainer = pl.Trainer(
        accelerator=cfg.trainer.parameters.accelerator,
        devices=cfg.trainer.parameters.devices,
        precision=cfg.trainer.parameters.precision,
        logger=[logger, csv_logger],
        enable_checkpointing=False,  # No checkpointing needed for evaluation
        enable_progress_bar=True,
        deterministic=True,
    )
    
    # Run evaluation
    print("\nRunning evaluation on test set...", flush=True)
    trainer.test(model=model, dataloaders=test_loader)
    
    # Save evaluation info to logs directory
    original_cwd = hydra.utils.get_original_cwd()
    version = logger.version if logger.version is not None else ""
    if version:
        log_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
    else:
        log_dir = os.path.join(logger.save_dir, logger.name)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Save evaluation config to log directory
    eval_info_path = os.path.join(log_dir, "evaluation_info.txt")
    with open(eval_info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION INFO\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test data base path: {cfg.data.base_path}\n")
        f.write(f"Test x_path: {cfg.data.path.test.x_path}\n")
        f.write(f"Test y_path: {cfg.data.path.test.y_path}\n")
        f.write(f"Batch size: {cfg.data.loader.batch_size}\n")
        f.write(f"Number of test batches: {len(test_loader)}\n")
        f.write(f"Seed: {seed}\n")
    print(f"Saved evaluation info to {eval_info_path}", flush=True)
    
    print("\n✓ Model evaluation pipeline completed!")


def train_regression_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 6: Train and evaluate the regression model.
    
    This pipeline:
    - Loads train/test features (HR, HRV) using utils functions
    - Trains logistic regression model with feature selection
    - Evaluates on test set with metrics and plots
    """
    print("=" * 80)
    print("PIPELINE 5: REGRESSION MODEL TRAINING")
    print("=" * 80)
    
    # Extract paths from config
    train_paths = cfg.data.path.train
    test_paths = cfg.data.path.test
    
    # Load training features and labels
    print("\nLoading training data...", flush=True)
    train_features, train_labels = load_train_biomarkers_and_labels(
        hr_path=train_paths.hr_path,
        hrv_path=train_paths.hrv_path,
        y_path=train_paths.y_path
    )
    print(f"Training data loaded: {len(train_features)} samples")
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels distribution:\n{train_labels.value_counts()}")
    
    # Load test features and labels
    print("\nLoading test data...", flush=True)
    test_features, test_labels = load_test_biomarkers_and_labels(
        hr_path=test_paths.hr_path,
        hrv_path=test_paths.hrv_path,
        y_path=test_paths.y_path
    )
    print(f"Test data loaded: {len(test_features)} samples")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels distribution:\n{test_labels.value_counts()}")
    
    # Initialize regression model
    print("\nInitializing regression model...", flush=True)
    model = Regression(
        n_feats_to_select=cfg.model.n_feats_to_select,
        reg_model=cfg.model.reg_model,
        get_summary=cfg.model.get_summary,
        confusion_matrix=cfg.model.confusion_matrix,
        classification_report=cfg.model.classification_report,
        aus_score=cfg.model.aus_score,
        plot_roc=cfg.model.plot_roc
    )
    
    # Train the model
    print("\nTraining regression model...", flush=True)
    model.train(train_features, train_labels)
    
    # Test the model
    print("\nTesting regression model...", flush=True)
    model.test(test_features, test_labels)
    
    print("\n✓ Regression model training pipeline completed!")


def train_svm_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 7: Train and evaluate the SVM model.
    
    This pipeline:
    - Loads train/test features (HR, HRV) using utils functions
    - Trains SVM model with feature selection
    - Evaluates on test set with same metrics as ResNet (accuracy, f1, specificity, AUC, sensitivity, PPV)
    """
    print("=" * 80)
    print("PIPELINE 7: SVM MODEL TRAINING")
    print("=" * 80)
    
    # Extract paths from config
    train_paths = cfg.data.path.train
    test_paths = cfg.data.path.test
    
    # Load training features and labels
    print("\nLoading training data...", flush=True)
    train_features, train_labels = load_train_biomarkers_and_labels(
        hr_path=train_paths.hr_path,
        hrv_path=train_paths.hrv_path,
        y_path=train_paths.y_path
    )
    print(f"Training data loaded: {len(train_features)} samples")
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels distribution:\n{train_labels.value_counts()}")
    
    # Load test features and labels
    print("\nLoading test data...", flush=True)
    test_features, test_labels = load_test_biomarkers_and_labels(
        hr_path=test_paths.hr_path,
        hrv_path=test_paths.hrv_path,
        y_path=test_paths.y_path
    )
    print(f"Test data loaded: {len(test_features)} samples")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels distribution:\n{test_labels.value_counts()}")
    
    # Initialize SVM model
    print("\nInitializing SVM model...", flush=True)
    model = SVM(
        n_feats_to_select=cfg.model.n_feats_to_select,
        kernel=cfg.model.kernel,
        C=cfg.model.C,
        gamma=cfg.model.gamma,
        class_weight=cfg.model.class_weight,
        get_summary=cfg.model.get_summary,
        confusion_matrix=cfg.model.confusion_matrix,
        classification_report=cfg.model.classification_report,
        aus_score=cfg.model.aus_score,
        plot_roc=cfg.model.plot_roc
    )
    
    # Train the model
    print("\nTraining SVM model...", flush=True)
    model.train(train_features, train_labels)
    
    # Test the model
    print("\nTesting SVM model...", flush=True)
    model.test(test_features, test_labels)
    
    print("\n✓ SVM model training pipeline completed!")


def train_random_forest_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline 8: Train and evaluate the Random Forest model.
    
    This pipeline:
    - Loads train/test features (HR, HRV) using utils functions
    - Trains Random Forest model with feature selection
    - Evaluates on test set with same metrics as ResNet (accuracy, f1, specificity, AUC, sensitivity, PPV)
    """
    print("=" * 80)
    print("PIPELINE 8: RANDOM FOREST MODEL TRAINING")
    print("=" * 80)
    
    # Extract paths from config
    train_paths = cfg.data.path.train
    test_paths = cfg.data.path.test
    
    # Load training features and labels
    print("\nLoading training data...", flush=True)
    train_features, train_labels = load_train_biomarkers_and_labels(
        hr_path=train_paths.hr_path,
        hrv_path=train_paths.hrv_path,
        y_path=train_paths.y_path
    )
    print(f"Training data loaded: {len(train_features)} samples")
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels distribution:\n{train_labels.value_counts()}")
    
    # Load test features and labels
    print("\nLoading test data...", flush=True)
    test_features, test_labels = load_test_biomarkers_and_labels(
        hr_path=test_paths.hr_path,
        hrv_path=test_paths.hrv_path,
        y_path=test_paths.y_path
    )
    print(f"Test data loaded: {len(test_features)} samples")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test labels distribution:\n{test_labels.value_counts()}")
    
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
        plot_roc=cfg.model.plot_roc
    )
    
    # Train the model
    print("\nTraining Random Forest model...", flush=True)
    model.train(train_features, train_labels)
    
    # Test the model
    print("\nTesting Random Forest model...", flush=True)
    model.test(test_features, test_labels)
    
    print("\n✓ Random Forest model training pipeline completed!")


def main():
    """
    Main entry point that routes to the appropriate pipeline.
    
    Usage:
        python main.py build_dataset    # Run dataset building pipeline
        python main.py preprocess       # Run preprocessing pipeline
        python main.py split            # Run data splitting pipeline
        python main.py domain_normalize # Run domain normalization pipeline
        python main.py train            # Run training pipeline
        python main.py train_regression # Run regression training pipeline
        python main.py train_svm        # Run SVM training pipeline
        python main.py train_random_forest # Run Random Forest training pipeline
    """
    if len(sys.argv) < 2:
        print("Please specify a pipeline to run.")
        print("\nAvailable pipelines:")
        print("  - build_dataset / build    : Build dataset with biomarkers")
        print("  - preprocess               : Preprocess signals")
        print("  - split                    : Split data into train/val/test")
        print("  - domain_normalize         : Normalize PPG by domain (Cathlab vs Theatre)")
        print("  - train                    : Train the classification model")
        print("  - train_regression         : Train the regression model")
        print("  - train_svm                : Train the SVM model")
        print("  - train_random_forest      : Train the Random Forest model")
        print("  - finetune / ft            : Finetune a pretrained model on new dataset")
        print("  - evaluate / eval / test   : Evaluate a trained model on test data")
        print("\nUsage: python main.py <pipeline_name>")
        sys.exit(1)
    
    pipeline_arg = sys.argv[1].lower()
    
    # Map command line argument to config and function
    pipeline_configs = {
        "build_dataset": ("extract_biomarkers_config", build_dataset_pipeline),
        "build": ("extract_biomarkers_config", build_dataset_pipeline),
        "extract": ("extract_biomarkers_config", build_dataset_pipeline),
        "preprocess": ("preprocess_config", preprocess_signals_pipeline),
        "preprocess_signals": ("preprocess_config", preprocess_signals_pipeline),
        "split": ("split_data_config", split_data_pipeline),
        "split_data": ("split_data_config", split_data_pipeline),
        "domain_normalize": ("domain_normalize_config", domain_normalize_pipeline),
        "domain_norm": ("domain_normalize_config", domain_normalize_pipeline),
        "normalize": ("domain_normalize_config", domain_normalize_pipeline),
        "train": ("train", train_model_pipeline),
        "train_model": ("train", train_model_pipeline),
        "train_regression": ("train_reg_config", train_regression_pipeline),
        "regression": ("train_reg_config", train_regression_pipeline),
        "train_svm": ("train_svm_config", train_svm_pipeline),
        "svm": ("train_svm_config", train_svm_pipeline),
        "train_random_forest": ("train_rf_config", train_random_forest_pipeline),
        "random_forest": ("train_rf_config", train_random_forest_pipeline),
        "rf": ("train_rf_config", train_random_forest_pipeline),
        "finetune": ("finetune", finetune_model_pipeline),
        "fine_tune": ("finetune", finetune_model_pipeline),
        "ft": ("finetune", finetune_model_pipeline),
        "evaluate": ("evaluate", evaluate_model_pipeline),
        "eval": ("evaluate", evaluate_model_pipeline),
    }
    
    if pipeline_arg not in pipeline_configs:
        print(f"Unknown pipeline: {pipeline_arg}")
        print("\nAvailable pipelines:")
        print("  - build_dataset / build    : Build dataset with biomarkers")
        print("  - preprocess               : Preprocess signals")
        print("  - split                    : Split data into train/val/test")
        print("  - domain_normalize         : Normalize PPG by domain (Cathlab vs Theatre)")
        print("  - train                    : Train the classification model")
        print("  - train_regression         : Train the regression model")
        print("  - train_svm                : Train the SVM model")
        print("  - train_random_forest      : Train the Random Forest model")
        print("  - finetune / ft            : Finetune a pretrained model on new dataset")
        print("  - evaluate / eval / test   : Evaluate a trained model on test data")
        sys.exit(1)
    
    config_name, pipeline_func = pipeline_configs[pipeline_arg]
    
    # Use Hydra to load config and run pipeline
    # config_path must be relative to the current working directory
    config_path = "config"
    
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
        pipeline_func(cfg)


if __name__ == '__main__':
    main()
