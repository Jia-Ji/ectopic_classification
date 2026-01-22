import os
import numpy as np
import json
from typing import Dict, Tuple
from omegaconf import DictConfig


class DomainNormalizer:
    """
    Domain-based normalization for PPG signals.
    Normalizes PPG data based on data source (Cathlab vs Cardiac Theatre).
    Computes domain statistics from training set only.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize DomainNormalizer with configuration from config file.
        
        Args:
            config: DictConfig containing normalization parameters
        """
        self.config = config
        self.normalize_cfg = config.domain_normalize
        
        # Load parameters from config
        self.data_dir = self.normalize_cfg.get('data_dir')
        self.stats_output_path = self.normalize_cfg.get('stats_output_path')
        
        # Domain prefixes
        self.cathlab_prefix = self.normalize_cfg.get('cathlab_prefix', 'ID')
        self.theatre_prefix = self.normalize_cfg.get('theatre_prefix', 'Theatre')
        
        # Validate required paths
        if not self.data_dir:
            raise ValueError("input_dir or data_dir must be specified in config.domain_normalize")
        if not self.stats_output_path:
            raise ValueError("stats_output_path must be specified in config.domain_normalize")
    
    def identify_domain(self, id0: str) -> str:
        """
        Identify domain based on ID0 prefix.
        
        Args:
            id0: Patient ID string
            
        Returns:
            Domain name: 'cathlab' or 'theatre'
        """
        id0_str = str(id0)
        if id0_str.startswith(self.cathlab_prefix):
            return 'cathlab'
        elif id0_str.startswith(self.theatre_prefix):
            return 'theatre'
        else:
            # Default to cathlab if prefix doesn't match
            print(f"Warning: ID0 '{id0_str}' doesn't match expected prefixes. Defaulting to 'cathlab'.")
            return 'cathlab'
    
    def load_split_data(self, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ID0 and PPG data for a given split.
        
        Args:
            split_name: Name of the split ('train', 'val', or 'test')
            
        Returns:
            Tuple of (id0_array, ppg_array)
        """
        id0_path = os.path.join(self.data_dir, f'id0_{split_name}.npy')
        ppg_path = os.path.join(self.data_dir, f'ppg_{split_name}.npy')
        
        if not os.path.exists(id0_path):
            raise FileNotFoundError(f"ID0 file not found: {id0_path}")
        if not os.path.exists(ppg_path):
            raise FileNotFoundError(f"PPG file not found: {ppg_path}")
        
        id0 = np.load(id0_path, allow_pickle=True)
        ppg = np.load(ppg_path, allow_pickle=True)
        
        if len(id0) != len(ppg):
            raise ValueError(f"Length mismatch: ID0 has {len(id0)} samples, PPG has {len(ppg)} samples")
        
        print(f"  Loaded {split_name}: {len(id0)} samples")
        return id0, ppg
    
    def compute_domain_stats(self, id0_train: np.ndarray, ppg_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute domain statistics (mean and std) from training set only.
        
        Args:
            id0_train: Array of patient IDs from training set
            ppg_train: Array of PPG signals from training set
            
        Returns:
            Dictionary with domain stats: {'cathlab': {'mean': float, 'std': float}, 
                                          'theatre': {'mean': float, 'std': float}}
        """
        print("\n=== Computing Domain Statistics from Training Set ===")
        
        # Group PPG segments by domain
        cathlab_segments = []
        theatre_segments = []
        
        for i in range(len(id0_train)):
            domain = self.identify_domain(id0_train[i])
            ppg_signal = np.asarray(ppg_train[i])
            
            # Flatten the signal for statistics computation
            if ppg_signal.ndim > 1:
                ppg_signal = ppg_signal.flatten()
            
            if domain == 'cathlab':
                cathlab_segments.append(ppg_signal)
            elif domain == 'theatre':
                theatre_segments.append(ppg_signal)
        
        # Compute statistics for each domain
        domain_stats = {}
        
        if len(cathlab_segments) > 0:
            # Concatenate all cathlab segments
            all_cathlab = np.concatenate(cathlab_segments)
            mean_cathlab = np.mean(all_cathlab)
            std_cathlab = np.std(all_cathlab)
            domain_stats['cathlab'] = {
                'mean': float(mean_cathlab),
                'std': float(std_cathlab),
                'num_segments': len(cathlab_segments)
            }
            print(f"  Cathlab: mean={mean_cathlab:.6f}, std={std_cathlab:.6f}, segments={len(cathlab_segments)}")
        else:
            print("  Warning: No cathlab segments found in training set")
            domain_stats['cathlab'] = {'mean': 0.0, 'std': 1.0, 'num_segments': 0}
        
        if len(theatre_segments) > 0:
            # Concatenate all theatre segments
            all_theatre = np.concatenate(theatre_segments)
            mean_theatre = np.mean(all_theatre)
            std_theatre = np.std(all_theatre)
            domain_stats['theatre'] = {
                'mean': float(mean_theatre),
                'std': float(std_theatre),
                'num_segments': len(theatre_segments)
            }
            print(f"  Theatre: mean={mean_theatre:.6f}, std={std_theatre:.6f}, segments={len(theatre_segments)}")
        else:
            print("  Warning: No theatre segments found in training set")
            domain_stats['theatre'] = {'mean': 0.0, 'std': 1.0, 'num_segments': 0}
        
        return domain_stats
    
    def normalize_by_domain(self, id0: np.ndarray, ppg: np.ndarray, 
                            domain_stats: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Normalize PPG signals by domain statistics.
        
        Args:
            id0: Array of patient IDs
            ppg: Array of PPG signals
            domain_stats: Dictionary with domain statistics
            
        Returns:
            Normalized PPG array
        """
        normalized_ppg = []
        
        for i in range(len(id0)):
            domain = self.identify_domain(id0[i])
            ppg_signal = np.asarray(ppg[i]).copy()
            
            # Get domain statistics
            stats = domain_stats[domain]
            mean_d = stats['mean']
            std_d = stats['std']
            
            # Avoid division by zero
            if std_d == 0:
                std_d = 1.0
            
            # Normalize: (x - mean_d) / std_d
            if ppg_signal.ndim > 1:
                # Handle multi-dimensional arrays (e.g., (1, signal_length))
                normalized_signal = (ppg_signal - mean_d) / std_d
            else:
                normalized_signal = (ppg_signal - mean_d) / std_d
            
            normalized_ppg.append(normalized_signal)
        
        return np.array(normalized_ppg, dtype=object)
    
    def save_domain_stats(self, domain_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Save domain statistics to JSON file.
        
        Args:
            domain_stats: Dictionary with domain statistics
        """
        os.makedirs(os.path.dirname(self.stats_output_path) if os.path.dirname(self.stats_output_path) else '.', exist_ok=True)
        
        with open(self.stats_output_path, 'w') as f:
            json.dump(domain_stats, f, indent=2)
        
        print(f"\n✓ Saved domain statistics to {self.stats_output_path}")
    
    def save_normalized_data(self, split_name: str, ppg_normalized: np.ndarray) -> None:
        """
        Save normalized PPG data.
        
        Args:
            split_name: Name of the split ('train', 'val', or 'test')
            ppg_normalized: Normalized PPG array
        """
        os.makedirs(self.data_dir, exist_ok=True)
        
        output_path = os.path.join(self.data_dir, f'ppg_{split_name}_normalized.npy')
        np.save(output_path, ppg_normalized)
        print(f"  ✓ Saved normalized {split_name} PPG to {output_path}")
    
    def run(self) -> None:
        """
        Run the complete domain normalization pipeline:
        1. Load training data
        2. Compute domain statistics from training set
        3. Save domain statistics
        4. Normalize all splits (train, val, test) using domain stats
        5. Save normalized data
        """
        print("=" * 80)
        print("DOMAIN-BASED PPG NORMALIZATION PIPELINE")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Cathlab prefix: '{self.cathlab_prefix}'")
        print(f"Theatre prefix: '{self.theatre_prefix}'")
        print("=" * 80)
        
        # Step 1: Load training data
        print("\n=== Step 1: Loading Training Data ===")
        id0_train, ppg_train = self.load_split_data('train')
        
        # Step 2: Compute domain statistics from training set
        print("\n=== Step 2: Computing Domain Statistics ===")
        domain_stats = self.compute_domain_stats(id0_train, ppg_train)
        
        # Step 3: Save domain statistics
        print("\n=== Step 3: Saving Domain Statistics ===")
        self.save_domain_stats(domain_stats)
        
        # Step 4: Normalize all splits
        print("\n=== Step 4: Normalizing All Splits ===")
        splits = ['train', 'val', 'test']
        
        for split_name in splits:
            try:
                print(f"\nProcessing {split_name} split...")
                id0_split, ppg_split = self.load_split_data(split_name)
                
                # Normalize by domain
                ppg_normalized = self.normalize_by_domain(id0_split, ppg_split, domain_stats)
                
                # Save normalized data
                self.save_normalized_data(split_name, ppg_normalized)
                
                # Print domain distribution for this split
                domains = [self.identify_domain(id0_split[i]) for i in range(len(id0_split))]
                cathlab_count = domains.count('cathlab')
                theatre_count = domains.count('theatre')
                print(f"  Domain distribution: Cathlab={cathlab_count}, Theatre={theatre_count}")
                
            except FileNotFoundError as e:
                print(f"  Warning: {split_name} split not found, skipping: {e}")
                continue
        
        print("\n" + "=" * 80)
        print("Domain normalization complete!")
        print("=" * 80)
        print(f"\nNormalized data saved to: {self.data_dir}")
        print(f"Domain statistics saved to: {self.stats_output_path}")

