import numpy as np
from typing import Callable, List, Optional, Sequence, Union


def set_augmentations_seed(seed: int) -> None:
    """Set NumPy RNG seed used by augmentation transforms for reproducibility."""
    np.random.seed(int(seed))

class SelectiveAugmentation:
    """Apply augmentation only to specific classes (e.g., ectopic segments)"""
    def __init__(self, transform, target_classes=[1], p=1.0):
        """
        Args:
            transform: The augmentation transform to apply
            target_classes: List of class labels to augment (e.g., [1] for ectopic)
            p: Probability of applying augmentation to target classes
        """
        self.transform = transform
        self.target_classes = target_classes
        self.p = p
    
    def __call__(self, x: np.ndarray, y: int = None) -> np.ndarray:
        # If no label provided, apply augmentation (for backward compatibility)
        if y is None:
            return self.transform(x)
        
        # Only apply augmentation to target classes
        if y in self.target_classes and np.random.rand() < self.p:
            return self.transform(x)
        else:
            return x

class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = [t for t in transforms if t is not None]
    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x

def _map_over_samples_1d(x: np.ndarray, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a 1D transform per-sample along the last dimension.
    Supports inputs shaped as (..., L) and preserves leading dimensions.
    """
    x_arr = np.asarray(x)
    if x_arr.ndim == 1:
        return func(x_arr)
    lead_shape = x_arr.shape[:-1]
    length = x_arr.shape[-1]
    flat = x_arr.reshape(-1, length)
    out = np.stack([func(row) for row in flat], axis=0)
    return out.reshape(*lead_shape, length)

class AmplitudeScaling:
    def __init__(self, scale_min=0.8, scale_max=1.2, p=0.5):
        self.scale_min, self.scale_max, self.p = scale_min, scale_max, p
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p: return x
        s = np.random.uniform(self.scale_min, self.scale_max)
        return (x * s).astype(x.dtype)

class BaselineWander:
    def __init__(self, freq_hz=(0.05, 0.5), amp=(0.01, 0.05), p=0.5, fs=240):
        self.freq_hz, self.amp, self.p, self.fs = freq_hz, amp, p, fs
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p: return x
        n = x.shape[-1]
        f = np.random.uniform(*self.freq_hz)
        a = np.random.uniform(*self.amp)
        t = np.arange(n) / self.fs
        return (x + a * np.sin(2 * np.pi * f * t)).astype(x.dtype)

class AdditiveGaussianNoise:
    def __init__(self, std_frac=(0.005, 0.02), p=0.5):
        self.std_frac, self.p = std_frac, p
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p: return x
        std = np.random.uniform(*self.std_frac) * (np.std(x) + 1e-8)
        return x + np.random.normal(0.0, std, size=x.shape).astype(x.dtype)

class RandomDropouts:
    def __init__(self, max_frac=0.05, p=0.5):
        self.max_frac, self.p = max_frac, p
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p: return x
        n = x.shape[-1]
        k = int(np.random.uniform(0.0, self.max_frac) * n)
        if k <= 0: return x
        idx = np.random.choice(n, size=k, replace=False)
        x = x.copy()
        x[..., idx] = 0.0
        return x

class MotionArtifacts:
    def __init__(self, max_segments=3, seg_len_frac=(0.01, 0.05), amp_frac=(0.5, 2.0), p=0.5):
        self.max_segments, self.seg_len_frac, self.amp_frac, self.p = max_segments, seg_len_frac, amp_frac, p
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p: return x
        n = x.shape[-1]
        x_aug = x.copy()
        num_segments = np.random.randint(1, self.max_segments + 1)
        for _ in range(num_segments):
            L = int(np.random.uniform(*self.seg_len_frac) * n)
            if L <= 0: continue
            start = np.random.randint(0, max(1, n - L))
            end = start + L
            a = np.random.uniform(*self.amp_frac) * (np.std(x) + 1e-8)
            jitter = a * (2 * np.random.rand(end - start) - 1)
            x_aug[..., start:end] = x_aug[..., start:end] + jitter.astype(x_aug.dtype)
        return x_aug

class TimeScaling:
    def __init__(self, scale_min=0.9, scale_max=1.1, p=0.5, mode="linear"):
        self.scale_min, self.scale_max, self.p, self.mode = scale_min, scale_max, p, mode
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p: return x

        def _scale_1d(row: np.ndarray) -> np.ndarray:
            s = np.random.uniform(self.scale_min, self.scale_max)
            n = row.shape[-1]
            grid = np.linspace(0, n - 1, int(n / s))
            scaled = np.interp(grid, np.arange(n), row.astype(np.float32))
            if scaled.shape[-1] < n:
                out = np.zeros(n, dtype=scaled.dtype)
                out[:scaled.shape[-1]] = scaled
                return out.astype(row.dtype)
            return scaled[:n].astype(row.dtype)

        return _map_over_samples_1d(x, _scale_1d)
