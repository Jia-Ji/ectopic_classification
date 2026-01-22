import numpy as np
import torch
from torch.utils.data import Dataset
#from data.augmentations import SelectiveAugmentation, Compose


def apply_mask_dropout(batch_tensor, p=0.3):
    """
    Apply mask dropout to a batch tensor.
    
    Args:
        batch_tensor: shape (B, 5, L) or (5, L) for single sample
            channel 0 = raw PPG
            channels 1-4 = FP masks (on, sp, dn, dp)
        p: probability of dropping ALL masks
    
    Returns:
        tuple: (result_tensor, was_dropped)
            result_tensor: Tensor/array with masks potentially zeroed out
            was_dropped: Boolean indicating if dropout was actually applied
    """
    # Make a copy to avoid modifying the original
    if isinstance(batch_tensor, np.ndarray):
        result = batch_tensor.copy()
    elif isinstance(batch_tensor, torch.Tensor):
        result = batch_tensor.clone()
    else:
        raise TypeError(f"Unsupported type: {type(batch_tensor)}")
    
    # Check if masks have any non-zero values before dropout
    if result.ndim == 2:
        # Single sample: (5, L)
        masks_before = result[1:, :]
        has_nonzero = np.any(masks_before != 0)
    elif result.ndim == 3:
        # Batch: (B, 5, L)
        masks_before = result[:, 1:, :]
        has_nonzero = np.any(masks_before != 0)
    else:
        raise ValueError(f"Unexpected tensor shape: {result.shape}")
    
    # Use numpy random for consistency (works with both numpy and torch)
    should_drop = np.random.rand() < p
    was_dropped = False
    
    if should_drop and has_nonzero:
        # Only drop if masks actually have values
        if result.ndim == 2:
            # Single sample: (5, L)
            result[1:, :] = 0  # zero-out all mask channels (channels 1-4)
        elif result.ndim == 3:
            # Batch: (B, 5, L)
            result[:, 1:, :] = 0  # zero-out all mask channels (channels 1-4)
        was_dropped = True
    
    return result, was_dropped


class TrainDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        fp_mask_path: str = None,
        sqi_path: str = None,
        sqi_sample_path: str = None,
        transform=None,
    ):
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        # Optional SQI array (shape: [N]) - scalar per sample
        self.sqi = None
        if sqi_path is not None:
            try:
                self.sqi = np.load(sqi_path, allow_pickle=True).astype(np.float32)
                if len(self.sqi) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi ({len(self.sqi)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi file not found at {sqi_path}. SQI will not be used.")
                self.sqi = None

        # Optional SQI sample array (shape: [N, T]) - temporal SQI per sample
        self.sqi_sample = None
        if sqi_sample_path is not None:
            try:
                self.sqi_sample = np.load(sqi_sample_path, allow_pickle=True)
                if len(self.sqi_sample) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi_sample ({len(self.sqi_sample)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi_sample file not found at {sqi_sample_path}. SQI sample will not be used.")
                self.sqi_sample = None

        self.fp_mask = None
        if fp_mask_path is not None:
            try:
                self.fp_mask = np.load(fp_mask_path, allow_pickle=True)
                if len(self.fp_mask) != len(self.x):
                    raise ValueError(f"Length mismatch: fp_mask ({len(self.fp_mask)}) != x ({len(self.x)})")
            except FileNotFoundError:
                print(f"Warning: fp_mask file not found at {fp_mask_path}. Will not concatenate fp_mask.")
                self.fp_mask = None
        self.transform = transform
    
    def __getitem__(self, index):
        x_get = self.x[index].astype(np.float32)
        if self.transform is not None:
            x_get = self.transform(x_get)
            x_get = x_get.astype(np.float32)
        
        # Concatenate fp_mask after transform (as requested)
        if self.fp_mask is not None:
            fp_mask_get = self.fp_mask[index].astype(np.float32)
            # Ensure x_get is 2D: (channels, signal_length)
            if x_get.ndim == 1:
                x_get = np.expand_dims(x_get, axis=0)
            elif x_get.ndim == 2:
                # If shape is (signal_length, channels), transpose to (channels, signal_length)
                if x_get.shape[0] > x_get.shape[1]:
                    x_get = x_get.T
            # fp_mask_get shape: (signal_length, 4), transpose to (4, signal_length)
            fp_mask_get = fp_mask_get.T  # (4, signal_length)
            x_get = np.concatenate([x_get, fp_mask_get], axis=0)  # (5, signal_length)
        else:
            # If fp_mask is not available, ensure x_get is 2D
            if x_get.ndim == 1:
                x_get = np.expand_dims(x_get, axis=0)
            elif x_get.ndim == 2:
                # If shape is (signal_length, channels), transpose to (channels, signal_length)
                if x_get.shape[0] > x_get.shape[1]:
                    x_get = x_get.T
        
        y_get = self.y[index].astype(np.int64)

        # Return both sqi (scalar) and sqi_sample (temporal) if available
        if self.sqi_sample is not None and self.sqi is not None:
            sqi_sample_get = self.sqi_sample[index].astype(np.float32)
            sqi_get = float(self.sqi[index])
            return x_get, y_get, sqi_get, sqi_sample_get
        elif self.sqi_sample is not None:
            sqi_sample_get = self.sqi_sample[index].astype(np.float32)
            return x_get, y_get, sqi_sample_get
        elif self.sqi is not None:
            sqi_get = float(self.sqi[index])
            return x_get, y_get, sqi_get

        return x_get, y_get
    
    def __len__(self):
        return len(self.y)

class ValidDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        fp_mask_path: str = None,
        sqi_path: str = None,
        sqi_sample_path: str = None,
        id0_path: str = None,
    ):
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)
        
        # Load id0 (patient IDs) for per-subject metrics
        self.id0 = None
        if id0_path is not None:
            try:
                self.id0 = np.load(id0_path, allow_pickle=True)
                if len(self.id0) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: id0 ({len(self.id0)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: id0 file not found at {id0_path}. Per-subject metrics will not be available.")
                self.id0 = None

        # Optional SQI array (shape: [N]) - scalar per sample
        self.sqi = None
        if sqi_path is not None:
            try:
                self.sqi = np.load(sqi_path, allow_pickle=True).astype(np.float32)
                if len(self.sqi) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi ({len(self.sqi)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi file not found at {sqi_path}. SQI will not be used.")
                self.sqi = None

        # Optional SQI sample array (shape: [N, T]) - temporal SQI per sample
        self.sqi_sample = None
        if sqi_sample_path is not None:
            try:
                self.sqi_sample = np.load(sqi_sample_path, allow_pickle=True)
                if len(self.sqi_sample) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi_sample ({len(self.sqi_sample)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi_sample file not found at {sqi_sample_path}. SQI sample will not be used.")
                self.sqi_sample = None

        self.fp_mask = None
        if fp_mask_path is not None:
            try:
                self.fp_mask = np.load(fp_mask_path, allow_pickle=True)
                if len(self.fp_mask) != len(self.x):
                    raise ValueError(f"Length mismatch: fp_mask ({len(self.fp_mask)}) != x ({len(self.x)})")
            except FileNotFoundError:
                print(f"Warning: fp_mask file not found at {fp_mask_path}. Will not concatenate fp_mask.")
                self.fp_mask = None
    
    def __getitem__(self, index):
        x_get = self.x[index].astype(np.float32)
        
        # Concatenate fp_mask with x to form 5-channel input
        if self.fp_mask is not None:
            fp_mask_get = self.fp_mask[index].astype(np.float32)
            # Ensure x_get is 2D: (channels, signal_length)
            if x_get.ndim == 1:
                x_get = np.expand_dims(x_get, axis=0)
            elif x_get.ndim == 2:
                # If shape is (signal_length, channels), transpose to (channels, signal_length)
                if x_get.shape[0] > x_get.shape[1]:
                    x_get = x_get.T
            # fp_mask_get shape: (signal_length, 4), transpose to (4, signal_length)
            fp_mask_get = fp_mask_get.T  # (4, signal_length)
            x_get = np.concatenate([x_get, fp_mask_get], axis=0)  # (5, signal_length)
        else:
            # If fp_mask is not available, ensure x_get is 2D
            if x_get.ndim == 1:
                x_get = np.expand_dims(x_get, axis=0)
            elif x_get.ndim == 2:
                # If shape is (signal_length, channels), transpose to (channels, signal_length)
                if x_get.shape[0] > x_get.shape[1]:
                    x_get = x_get.T
        
        y_get = self.y[index].astype(np.int64)
        
        # Get id0 if available
        id0_get = None
        if self.id0 is not None:
            id0_get = str(self.id0[index])

        # Return both sqi (scalar) and sqi_sample (temporal) if available
        # id0 is always returned last if available
        if self.sqi_sample is not None and self.sqi is not None:
            sqi_sample_get = self.sqi_sample[index].astype(np.float32)
            sqi_get = float(self.sqi[index])
            if id0_get is not None:
                return x_get, y_get, sqi_get, sqi_sample_get, id0_get
            return x_get, y_get, sqi_get, sqi_sample_get
        elif self.sqi_sample is not None:
            sqi_sample_get = self.sqi_sample[index].astype(np.float32)
            if id0_get is not None:
                return x_get, y_get, sqi_sample_get, id0_get
            return x_get, y_get, sqi_sample_get
        elif self.sqi is not None:
            sqi_get = float(self.sqi[index])
            if id0_get is not None:
                return x_get, y_get, sqi_get, id0_get
            return x_get, y_get, sqi_get
        
        if id0_get is not None:
            return x_get, y_get, id0_get

        return x_get, y_get
    
    def __len__(self):
        return len(self.y)

class TestDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        ecg_path: str = None,
        include_ecg: bool = False,
        filenames_path: str = None,
        fp_mask_path: str = None,
        sqi_path: str = None,
        sqi_sample_path: str = None,
    ):
        super().__init__()

        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        # Optional SQI array (shape: [N]) – mainly for analysis; not used directly in loss here
        self.sqi = None
        if sqi_path is not None:
            try:
                self.sqi = np.load(sqi_path, allow_pickle=True).astype(np.float32)
                if len(self.sqi) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi ({len(self.sqi)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi file not found at {sqi_path}. SQI will not be returned.")
                self.sqi = None

        # Optional SQI sample array (shape: [N, T]) - temporal SQI per sample
        self.sqi_sample = None
        if sqi_sample_path is not None:
            try:
                self.sqi_sample = np.load(sqi_sample_path, allow_pickle=True)
                if len(self.sqi_sample) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi_sample ({len(self.sqi_sample)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi_sample file not found at {sqi_sample_path}. SQI sample will not be returned.")
                self.sqi_sample = None

        self.ecg = None
        self.include_ecg = False

        if ecg_path is not None:
            self.ecg = np.load(ecg_path, allow_pickle=True)
            if len(self.ecg) != len(self.x):
                raise ValueError(
                    "Length mismatch between PPG and ECG test splits."
                )

        if include_ecg:
            if self.ecg is None:
                raise ValueError("include_ecg=True but ecg_path was not provided or failed to load.")
            self.include_ecg = True
        
        # Load filenames if provided
        self.filenames = None
        if filenames_path is not None:
            try:
                self.filenames = np.load(filenames_path, allow_pickle=True)
                if len(self.filenames) != len(self.x):
                    print(f"Warning: Length mismatch between filenames ({len(self.filenames)}) and data ({len(self.x)}). Filenames will not be returned.")
                    self.filenames = None
            except FileNotFoundError:
                print(f"Warning: Filenames file not found at {filenames_path}. Filenames will not be returned.")
                self.filenames = None
        
        # Load fp_mask if provided
        self.fp_mask = None
        if fp_mask_path is not None:
            try:
                self.fp_mask = np.load(fp_mask_path, allow_pickle=True)
                if len(self.fp_mask) != len(self.x):
                    raise ValueError(f"Length mismatch: fp_mask ({len(self.fp_mask)}) != x ({len(self.x)})")
            except FileNotFoundError:
                print(f"Warning: fp_mask file not found at {fp_mask_path}. Will not concatenate fp_mask.")
                self.fp_mask = None
    
    def __getitem__(self, index):
        x_get = self.x[index].astype(np.float32)
        
        # Concatenate fp_mask with x to form 5-channel input
        if self.fp_mask is not None:
            fp_mask_get = self.fp_mask[index].astype(np.float32)
            # Ensure x_get is 2D: (channels, signal_length)
            if x_get.ndim == 1:
                x_get = np.expand_dims(x_get, axis=0)
            elif x_get.ndim == 2:
                # If shape is (signal_length, channels), transpose to (channels, signal_length)
                if x_get.shape[0] > x_get.shape[1]:
                    x_get = x_get.T
            # fp_mask_get shape: (signal_length, 4), transpose to (4, signal_length)
            fp_mask_get = fp_mask_get.T  # (4, signal_length)
            x_get = np.concatenate([x_get, fp_mask_get], axis=0)  # (5, signal_length)
        else:
            # If fp_mask is not available, ensure x_get is 2D
            if x_get.ndim == 1:
                x_get = np.expand_dims(x_get, axis=0)
            elif x_get.ndim == 2:
                # If shape is (signal_length, channels), transpose to (channels, signal_length)
                if x_get.shape[0] > x_get.shape[1]:
                    x_get = x_get.T
        
        y_get = self.y[index].astype(np.int64)

        # Extract both sqi (scalar) and sqi_sample (temporal) if available
        sqi_get = None
        sqi_sample_get = None
        if self.sqi is not None:
            sqi_get = float(self.sqi[index])
        if self.sqi_sample is not None:
            sqi_sample_get = self.sqi_sample[index].astype(np.float32)

        # Build return tuples: (x, y, ..., sqi, sqi_sample)
        # sqi is always before sqi_sample, and both are at the end
        if self.include_ecg:
            ecg_get = self.ecg[index].astype(np.float32)
            if self.filenames is not None:
                if sqi_get is not None and sqi_sample_get is not None:
                    return x_get, y_get, ecg_get, self.filenames[index], sqi_get, sqi_sample_get
                elif sqi_get is not None:
                    return x_get, y_get, ecg_get, self.filenames[index], sqi_get
                elif sqi_sample_get is not None:
                    return x_get, y_get, ecg_get, self.filenames[index], sqi_sample_get
                return x_get, y_get, ecg_get, self.filenames[index]
            if sqi_get is not None and sqi_sample_get is not None:
                return x_get, y_get, ecg_get, sqi_get, sqi_sample_get
            elif sqi_get is not None:
                return x_get, y_get, ecg_get, sqi_get
            elif sqi_sample_get is not None:
                return x_get, y_get, ecg_get, sqi_sample_get
            return x_get, y_get, ecg_get

        if self.filenames is not None:
            if sqi_get is not None and sqi_sample_get is not None:
                return x_get, y_get, self.filenames[index], sqi_get, sqi_sample_get
            elif sqi_get is not None:
                return x_get, y_get, self.filenames[index], sqi_get
            elif sqi_sample_get is not None:
                return x_get, y_get, self.filenames[index], sqi_sample_get
            return x_get, y_get, self.filenames[index]
        
        if sqi_get is not None and sqi_sample_get is not None:
            return x_get, y_get, sqi_get, sqi_sample_get
        elif sqi_get is not None:
            return x_get, y_get, sqi_get
        elif sqi_sample_get is not None:
            return x_get, y_get, sqi_sample_get

        return x_get, y_get
    
    def __len__(self):
        return len(self.y)

class BalancedTrainDataset(Dataset):
    """Dataset that balances classes by augmenting minority classes (Ectopic and VT)"""
    def __init__(
        self,
        x_path: str,
        y_path: str,
        fp_mask_path: str = None,
        sqi_path: str = None,
        sqi_sample_path: str = None,
        id0_path: str = None,
        augmentation_transforms=None,
        target_ratio=1.0,
        mask_dropout_p=0.3,
    ):
        """
        Args:
            x_path: Path to features
            y_path: Path to labels
            fp_mask_path: Path to fp_mask file (optional)
            sqi_path: Path to SQI file (optional) - scalar per sample
            sqi_sample_path: Path to SQI sample file (optional) - temporal array per sample
            augmentation_transforms: List of augmentation transforms
            target_ratio: Target ratio of minority classes to majority class (1.0 = equal)
            mask_dropout_p: Probability of dropping all mask channels (default: 0.3)
        """
        super().__init__()
        
        self.x = np.load(x_path, allow_pickle=True)
        self.y = np.load(y_path, allow_pickle=True)

        # Optional SQI array (shape: [N]) - scalar per sample
        self.sqi = None
        if sqi_path is not None:
            try:
                self.sqi = np.load(sqi_path, allow_pickle=True).astype(np.float32)
                if len(self.sqi) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi ({len(self.sqi)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi file not found at {sqi_path}. SQI-based loss weighting will be disabled.")
                self.sqi = None

        # Optional SQI sample array (shape: [N, T]) - temporal SQI per sample
        self.sqi_sample = None
        if sqi_sample_path is not None:
            try:
                self.sqi_sample = np.load(sqi_sample_path, allow_pickle=True)
                if len(self.sqi_sample) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: sqi_sample ({len(self.sqi_sample)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: sqi_sample file not found at {sqi_sample_path}. SQI sample will not be used.")
                self.sqi_sample = None

        self.fp_mask = None
        if fp_mask_path is not None:
            try:
                self.fp_mask = np.load(fp_mask_path, allow_pickle=True)
                if len(self.fp_mask) != len(self.x):
                    raise ValueError(f"Length mismatch: fp_mask ({len(self.fp_mask)}) != x ({len(self.x)})")
            except FileNotFoundError:
                print(f"Warning: fp_mask file not found at {fp_mask_path}. Will not concatenate fp_mask.")
                self.fp_mask = None
        
        # Load id0 (patient IDs) for per-subject metrics
        self.id0 = None
        if id0_path is not None:
            try:
                self.id0 = np.load(id0_path, allow_pickle=True)
                if len(self.id0) != len(self.x):
                    raise ValueError(
                        f"Length mismatch: id0 ({len(self.id0)}) != x ({len(self.x)})"
                    )
            except FileNotFoundError:
                print(f"Warning: id0 file not found at {id0_path}. Per-subject metrics will not be available.")
                self.id0 = None
        
        self.augmentation_transforms = augmentation_transforms or []
        self.mask_dropout_p = mask_dropout_p
        
        # Debug: track mask dropout statistics
        self._mask_dropout_count = 0
        self._total_samples = 0
        
        # Calculate class distribution for 3 classes: Normal=0, Ectopic=1, VT=2
        self.normal_indices = np.where(self.y == 0)[0]
        self.ectopic_indices = np.where(self.y == 1)[0]
        self.vt_indices = np.where(self.y == 2)[0]
        
        self.n_normal = len(self.normal_indices)
        self.n_ectopic = len(self.ectopic_indices)
        self.n_vt = len(self.vt_indices)
        
        # Find the majority class size
        self.n_majority = max(self.n_normal, self.n_ectopic, self.n_vt)
        
        # Calculate how many augmented samples we need for each minority class
        target_ectopic = int(self.n_majority * target_ratio)
        target_vt = int(self.n_majority * target_ratio)
        
        self.n_augmentations_ectopic = max(0, target_ectopic - self.n_ectopic)
        self.n_augmentations_vt = max(0, target_vt - self.n_vt)
        self.n_augmentations_needed = self.n_augmentations_ectopic + self.n_augmentations_vt
        
        print(f"Original distribution: {self.n_normal} Normal, {self.n_ectopic} Ectopic, {self.n_vt} VT")
        print(f"Target ratio: {target_ratio} (relative to majority class: {self.n_majority})")
        print(f"Will generate {self.n_augmentations_ectopic} augmented Ectopic samples")
        print(f"Will generate {self.n_augmentations_vt} augmented VT samples")
        print(f"Total augmented samples: {self.n_augmentations_needed}")
    
    def __getitem__(self, index):
        if index < len(self.x):
            # Original sample - NO augmentation applied
            x_get = self.x[index].astype(np.float32)
            y_get = self.y[index].astype(np.int64)
            
            # Get id0 if available
            id0_get = None
            if self.id0 is not None:
                id0_get = str(self.id0[index])
            
            # Concatenate fp_mask (no augmentation on original samples)
            if self.fp_mask is not None:
                fp_mask_get = self.fp_mask[index].astype(np.float32)
                # Ensure x_get is 2D: (channels, signal_length)
                if x_get.ndim == 1:
                    x_get = np.expand_dims(x_get, axis=0)
                elif x_get.ndim == 2:
                    # If shape is (signal_length, channels), transpose to (channels, signal_length)
                    if x_get.shape[0] > x_get.shape[1]:
                        x_get = x_get.T
                # fp_mask_get shape: (signal_length, 4), transpose to (4, signal_length)
                fp_mask_get = fp_mask_get.T  # (4, signal_length)
                x_get = np.concatenate([x_get, fp_mask_get], axis=0)  # (5, signal_length)
                
                # Apply mask dropout
                x_get, was_dropped = apply_mask_dropout(x_get, p=self.mask_dropout_p)
                # Track statistics
                self._total_samples += 1
                if was_dropped:
                    self._mask_dropout_count += 1
                # Print statistics every 1000 samples (optional debug)
                if self._total_samples % 1000 == 0:
                    dropout_rate = self._mask_dropout_count / self._total_samples
                    print(f"Mask dropout stats: {self._mask_dropout_count}/{self._total_samples} samples dropped ({dropout_rate:.2%}), expected: {self.mask_dropout_p:.2%}")
            else:
                # If fp_mask is not available, ensure x_get is 2D
                if x_get.ndim == 1:
                    x_get = np.expand_dims(x_get, axis=0)
                elif x_get.ndim == 2:
                    # If shape is (signal_length, channels), transpose to (channels, signal_length)
                    if x_get.shape[0] > x_get.shape[1]:
                        x_get = x_get.T
            
            # Return both sqi (scalar) and sqi_sample (temporal) if available
            # id0 is always returned last if available
            if self.sqi_sample is not None and self.sqi is not None:
                sqi_sample_get = self.sqi_sample[index].astype(np.float32)
                sqi_get = float(self.sqi[index])
                if id0_get is not None:
                    return x_get, y_get, sqi_get, sqi_sample_get, id0_get
                return x_get, y_get, sqi_get, sqi_sample_get
            elif self.sqi_sample is not None:
                sqi_sample_get = self.sqi_sample[index].astype(np.float32)
                if id0_get is not None:
                    return x_get, y_get, sqi_sample_get, id0_get
                return x_get, y_get, sqi_sample_get
            elif self.sqi is not None:
                sqi_get = float(self.sqi[index])
                if id0_get is not None:
                    return x_get, y_get, sqi_get, id0_get
                return x_get, y_get, sqi_get

            if id0_get is not None:
                return x_get, y_get, id0_get
            return x_get, y_get
        else:
            # Augmented sample from minority classes
            aug_index = index - len(self.x)
            
            # Determine which class to augment and get the original index
            if aug_index < self.n_augmentations_ectopic:
                # Augmented Ectopic sample
                original_index = self.ectopic_indices[aug_index % len(self.ectopic_indices)]
                y_get = 1  # Ectopic class
            else:
                # Augmented VT sample
                vt_aug_index = aug_index - self.n_augmentations_ectopic
                original_index = self.vt_indices[vt_aug_index % len(self.vt_indices)]
                y_get = 2  # VT class
            
            x_get = self.x[original_index].astype(np.float32)
            
            # Get id0 from original index if available
            id0_get = None
            if self.id0 is not None:
                id0_get = str(self.id0[original_index])
            
            # Apply all augmentations
            for transform in self.augmentation_transforms:
                x_get = transform(x_get)
            x_get = x_get.astype(np.float32)
            
            # Concatenate fp_mask after transform (as requested)
            if self.fp_mask is not None:
                fp_mask_get = self.fp_mask[original_index].astype(np.float32)
                # Ensure x_get is 2D: (channels, signal_length)
                if x_get.ndim == 1:
                    x_get = np.expand_dims(x_get, axis=0)
                elif x_get.ndim == 2:
                    # If shape is (signal_length, channels), transpose to (channels, signal_length)
                    if x_get.shape[0] > x_get.shape[1]:
                        x_get = x_get.T
                # fp_mask_get shape: (signal_length, 4), transpose to (4, signal_length)
                fp_mask_get = fp_mask_get.T  # (4, signal_length)
                x_get = np.concatenate([x_get, fp_mask_get], axis=0)  # (5, signal_length)
                
                # Apply mask dropout
                x_get, was_dropped = apply_mask_dropout(x_get, p=self.mask_dropout_p)
                # Track statistics
                self._total_samples += 1
                if was_dropped:
                    self._mask_dropout_count += 1
                # Print statistics every 1000 samples (optional debug)
                if self._total_samples % 1000 == 0:
                    dropout_rate = self._mask_dropout_count / self._total_samples
                    print(f"Mask dropout stats: {self._mask_dropout_count}/{self._total_samples} samples dropped ({dropout_rate:.2%}), expected: {self.mask_dropout_p:.2%}")
            
            # For augmented ectopic samples, reuse original sample's SQI if available
            # Return both sqi (scalar) and sqi_sample (temporal) if available
            # id0 is always returned last if available
            if self.sqi_sample is not None and self.sqi is not None:
                sqi_sample_get = self.sqi_sample[original_index].astype(np.float32)
                sqi_get = float(self.sqi[original_index])
                if id0_get is not None:
                    return x_get, y_get, sqi_get, sqi_sample_get, id0_get
                return x_get, y_get, sqi_get, sqi_sample_get
            elif self.sqi_sample is not None:
                sqi_sample_get = self.sqi_sample[original_index].astype(np.float32)
                if id0_get is not None:
                    return x_get, y_get, sqi_sample_get, id0_get
                return x_get, y_get, sqi_sample_get
            elif self.sqi is not None:
                sqi_get = float(self.sqi[original_index])
                if id0_get is not None:
                    return x_get, y_get, sqi_get, id0_get
                return x_get, y_get, sqi_get

            if id0_get is not None:
                return x_get, y_get, id0_get
            return x_get, y_get
    
    def __len__(self):
        return len(self.x) + self.n_augmentations_needed
