import torch
import os
import numpy as np
import pandas as pd
import scipy.io
from typing import Dict, Any, Tuple
from data import TrainDataset, ValidDataset, TestDataset, BalancedTrainDataset
from data.augmentations import AmplitudeScaling, BaselineWander, AdditiveGaussianNoise, RandomDropouts, MotionArtifacts, TimeScaling, Compose

def create_train_data_loader(cfg):
    train_datasets = TrainDataset(**cfg.path.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_datasets, shuffle=True, **cfg.loader
    )

    valid_datasets = ValidDataset(**cfg.path.valid)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_datasets, shuffle=False, **cfg.loader
    )

    return train_dataloader, valid_dataloader

def create_test_data_loader(cfg):
    test_path_cfg = cfg.path.test

    dataset_kwargs = {
        "x_path": test_path_cfg.x_path,
        "y_path": test_path_cfg.y_path,
        "fp_mask_path": test_path_cfg.fp_mask_path,
        # Optional SQI for test (mainly for analysis / diagnostics)
        "sqi_path": getattr(test_path_cfg, "sqi_path", None),
        # Optional SQI sample (temporal array) for attention mechanism
        "sqi_sample_path": getattr(test_path_cfg, "sqi_sample_path", None),
    }

    ecg_path = getattr(test_path_cfg, "ecg_path", None)
    if ecg_path is not None:
        dataset_kwargs["ecg_path"] = ecg_path
        dataset_kwargs["include_ecg"] = getattr(test_path_cfg, "include_ecg", True)

    # Add filenames path if available
    filenames_path = getattr(test_path_cfg, "filenames_path", None)
    if filenames_path is not None:
        dataset_kwargs["filenames_path"] = filenames_path

    test_datasets = TestDataset(**dataset_kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        test_datasets, shuffle=False, **cfg.loader
    )

    return test_dataloader

def _build_ectopic_augmentations(aug_cfg):
    """Build augmentation pipeline specifically for ectopic segments"""
    if not getattr(aug_cfg, "enable", False):
        return []
    
    transforms = []
    
    if aug_cfg.amplitude_scaling.enable:
        transforms.append(AmplitudeScaling(**aug_cfg.amplitude_scaling.params))
    
    if aug_cfg.baseline_wander.enable:
        transforms.append(BaselineWander(**aug_cfg.baseline_wander.params))
    
    if aug_cfg.additive_gaussian_noise.enable:
        transforms.append(AdditiveGaussianNoise(**aug_cfg.additive_gaussian_noise.params))
    
    if aug_cfg.random_dropouts.enable:
        transforms.append(RandomDropouts(**aug_cfg.random_dropouts.params))
    
    if aug_cfg.motion_artifacts.enable:
        transforms.append(MotionArtifacts(**aug_cfg.motion_artifacts.params))
    
    if aug_cfg.time_scaling.enable:
        transforms.append(TimeScaling(**aug_cfg.time_scaling.params))
    
    return transforms

def create_balanced_train_data_loader(cfg):
    """Create balanced training data loader with ectopic augmentation"""
    
    # Build augmentation pipeline for ectopic segments
    augmentation_transforms = _build_ectopic_augmentations(cfg.augmentations) if hasattr(cfg, "augmentations") else []
    
    # Create balanced dataset
    train_datasets = BalancedTrainDataset(
        **cfg.path.train,
        augmentation_transforms=augmentation_transforms,
        target_ratio=getattr(cfg.augmentations, "target_ratio", 1.0),  # 1.0 = equal classes
        mask_dropout_p=getattr(cfg.augmentations, "mask_dropout_p", 0.3),  # Probability of dropping all masks
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_datasets, shuffle=True, **cfg.loader
    )

    # Construct id0_path from x_path (replace filename with id0_val.npy)
    valid_cfg = dict(cfg.path.valid)
    if 'x_path' in valid_cfg:
        import os
        x_path = valid_cfg['x_path']
        # Get directory and construct id0_path
        data_dir = os.path.dirname(x_path)
        id0_path = os.path.join(data_dir, 'id0_val.npy')
        if os.path.exists(id0_path):
            valid_cfg['id0_path'] = id0_path
        else:
            print(f"Warning: id0_val.npy not found at {id0_path}. Per-subject metrics will not be available.", flush=True)
    
    valid_datasets = ValidDataset(**valid_cfg)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_datasets, shuffle=False, **cfg.loader
    )

    return train_dataloader, valid_dataloader

def delete_empty_dirs(root_dir):
    """
    Recursively delete empty subdirectories under the given root directory.
    
    Args:
        root_dir (str): Path to the root directory to check.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Check if directory is empty (no files and no subdirectories)
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty directory: {dirpath}")
            except OSError as e:
                print(f"Error deleting {dirpath}: {e}")

def load_mat_file(path: str) -> Dict[str, Any]:
    """
    Load .mat files generated with either the legacy MAT format or MATLAB v7.3 (HDF5-based).
    """
    # First try v7.3 (HDF5) reader if available
    try:
        import h5py, hdf5storage  # v7.3
        if h5py.is_hdf5(path):
            return hdf5storage.loadmat(path)
    except Exception:
        # If the import fails or HDF5 load fails, we continue to legacy loader below
        pass
    # Fallback to legacy MATLAB reader; surface a helpful message if file is actually v7.3
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:
        raise RuntimeError(
            "The provided .mat file appears to be MATLAB v7.3 (HDF5). "
            "Please install the required packages and try again:\n"
            "  pip install h5py hdf5storage\n"
            "Alternatively, ensure your environment (pyPPG_source) has these packages available."
        ) from e

def load_pd_file(path: str) -> pd.DataFrame:
    """Load the pickle file containing PPG, ECG, labels, and metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    
    df = pd.read_pickle(path)
    print(f"Loaded dataframe with shape {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def extract_ppg_ecg(mat_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract PPG and ECG arrays from the loaded .mat data.

    The expected structure is a tensor stored under the key 'S' with shape (N, T, 2)
    or a similar arrangement where the third dimension indexes the PPG (0) and ECG (1) signals.
    """
    if "S" not in mat_data:
        raise KeyError("Key 'S' not found in the MATLAB file. Unable to extract signals.")

    signals = mat_data["S"]
    if signals.ndim != 3 or signals.shape[2] < 2:
        raise ValueError(
            "Unexpected shape for 'S'. Expected a 3-D array with at least two channels (PPG, ECG)."
        )

    ppg = np.transpose(signals[:, :, 0])
    ecg = np.transpose(signals[:, :, 1])

    return np.asarray(ppg), np.asarray(ecg)

def load_train_biomarkers_and_labels(hr_path, hrv_path, y_path):
    train_hr = np.load(hr_path)
    train_hrv = np.load(hrv_path)
    train_labels = np.load(y_path, allow_pickle=True)
    train_labels = pd.Series(train_labels)

    
    # Combine HR and HRV into a DataFrame
    train_features = pd.DataFrame({
        'HR': train_hr.flatten() if train_hr.ndim > 1 else train_hr,
        'HRV': train_hrv.flatten() if train_hrv.ndim > 1 else train_hrv
    })

    clean_train_features = train_features.dropna()
    clean_train_labels = train_labels.loc[clean_train_features.index]

    return clean_train_features, clean_train_labels

def load_test_biomarkers_and_labels(hr_path, hrv_path, y_path):
    test_hr = np.load(hr_path)
    test_hrv = np.load(hrv_path)
    test_labels = np.load(y_path, allow_pickle=True)
    test_labels = pd.Series(test_labels)

    # Combine HR and HRV into a DataFrame
    test_features = pd.DataFrame({
        'HR': test_hr.flatten() if test_hr.ndim > 1 else test_hr,
        'HRV': test_hrv.flatten() if test_hrv.ndim > 1 else test_hrv
    })

    clean_test_features = test_features.dropna()
    clean_test_labels = test_labels.loc[clean_test_features.index]

    return clean_test_features, clean_test_labels
