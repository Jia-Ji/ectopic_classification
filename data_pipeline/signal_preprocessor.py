import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import Dict, Any
from utils import load_pd_file
from omegaconf import DictConfig

class Preprocessor:
    """
    Preprocessing class for PPG/ECG signals.
    All configuration parameters are loaded from config file.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Preprocessor with configuration from config file.
        
        Args:
            config: DictConfig containing preprocessing parameters
        """
        self.config = config
        self.preprocess_cfg = config.preprocess
        
        # Load parameters from config
        self.fs = self.preprocess_cfg.get('fs', 240)
        self.bandpass_low = self.preprocess_cfg.get('bandpass_low', 0.3)
        self.bandpass_high = self.preprocess_cfg.get('bandpass_high', 8.0)
        self.bandpass_order = self.preprocess_cfg.get('bandpass_order', 4)
        self.iqr_k = self.preprocess_cfg.get('iqr_k', 3.0)
        
        # Paths from config
        self.input_path = self.preprocess_cfg.get('input_path')
        self.output_path = self.preprocess_cfg.get('output_path')
        
        # Validate required paths
        if not self.input_path:
            raise ValueError("input_path must be specified in config.preprocess")
        if not self.output_path:
            raise ValueError("output_path must be specified in config.preprocess")
    
    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply a bandpass filter to a signal.
        
        Args:
            signal: Input signal array
        
        Returns:
            Filtered signal array
        """
        nyquist = self.fs / 2.0
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        
        # Design Butterworth bandpass filter
        b, a = butter(self.bandpass_order, [low, high], btype='band')
        
        # Apply zero-phase filtering
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def clip_outliers_iqr(self, signal: np.ndarray) -> np.ndarray:
        """
        Clip extreme outliers using IQR (Interquartile Range) method.
        
        Args:
            signal: Input signal array
        
        Returns:
            Clipped signal array
        """
        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_k * iqr
        upper_bound = q3 + self.iqr_k * iqr
        
        clipped_signal = np.clip(signal, lower_bound, upper_bound)
        
        return clipped_signal
    
    # def normalize_per_segment(self, df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    #     """
    #     Normalize each signal segment individually using per-segment z-score normalization.
    #     Each segment is normalized using its own mean and standard deviation.
        
    #     Args:
    #         df: DataFrame with signals
    #         signal_column: Name of the column containing signals ('PPG' or 'ECG')
        
    #     Returns:
    #         DataFrame with normalized signals
    #     """
    #     df = df.copy()
        
    #     # Normalize each segment individually
    #     normalized_signals = []
        
    #     for idx in df.index:
    #         signal = np.asarray(df.loc[idx, signal_column])
            
    #         # Calculate mean and standard deviation for this segment
    #         mean_val = np.mean(signal)
    #         std_val = np.std(signal)
            
    #         # Avoid division by zero
    #         if std_val == 0:
    #             std_val = 1.0
            
    #         # Normalize this segment using its own statistics
    #         normalized_signal = (signal - mean_val) / std_val
    #         normalized_signals.append((idx, normalized_signal))
        
    #     # Update the dataframe with normalized signals
    #     for idx, normalized_signal in normalized_signals:
    #         df.at[idx, signal_column] = normalized_signal
        
        return df
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the dataframe:
        1. Bandpass filter
        2. Clip extreme outliers (IQR method)
        3. Per-segment z-score normalization (normalize each segment individually)
        
        Args:
            df: Input dataframe with PPG, ECG, and ID0 columns
        
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        print("\n=== Step 1: Bandpass Filtering ===")
        print(f"Filter range: {self.bandpass_low}-{self.bandpass_high} Hz")
        print(f"Filter order: {self.bandpass_order}")
        
        # Apply bandpass filter to PPG
        print("Filtering PPG signals...")
        ppg_filtered = []
        for idx in df.index:
            ppg_signal = np.asarray(df.loc[idx, 'PPG'])
            filtered_ppg = self.bandpass_filter(ppg_signal)
            ppg_filtered.append(filtered_ppg)
        df['PPG'] = ppg_filtered
        print(f"✓ Filtered {len(ppg_filtered)} PPG signals")
        
        # # Apply bandpass filter to ECG
        # print("Filtering ECG signals...")
        # ecg_filtered = []
        # for idx in df.index:
        #     ecg_signal = np.asarray(df.loc[idx, 'ECG'])
        #     filtered_ecg = self.bandpass_filter(ecg_signal)
        #     ecg_filtered.append(filtered_ecg)
        # df['ECG'] = ecg_filtered
        # print(f"✓ Filtered {len(ecg_filtered)} ECG signals")
        
        print(f"\n=== Step 2: Clipping Extreme Outliers (IQR, k={self.iqr_k}) ===")
        # Clip outliers in PPG
        print("Clipping outliers in PPG signals...")
        ppg_clipped = []
        for idx in df.index:
            ppg_signal = np.asarray(df.loc[idx, 'PPG'])
            clipped_ppg = self.clip_outliers_iqr(ppg_signal)
            ppg_clipped.append(clipped_ppg)
        df['PPG'] = ppg_clipped
        print(f"✓ Clipped {len(ppg_clipped)} PPG signals")
        
        # Clip outliers in ECG
        # print("Clipping outliers in ECG signals...")
        # ecg_clipped = []
        # for idx in df.index:
        #     ecg_signal = np.asarray(df.loc[idx, 'ECG'])
        #     clipped_ecg = self.clip_outliers_iqr(ecg_signal)
        #     ecg_clipped.append(clipped_ecg)
        # df['ECG'] = ecg_clipped
        # print(f"✓ Clipped {len(ecg_clipped)} ECG signals")
        
        # print("\n=== Step 3: Per-segment Z-score Normalization ===")
        # # Normalize PPG per segment
        # print("Normalizing PPG signals per segment...")
        # df = self.normalize_per_segment(df, 'PPG')
        # print(f"✓ Normalized PPG signals")
        
        # # Normalize ECG by ID0
        # print("Normalizing ECG signals by ID0...")
        # df = self.normalize_by_id0(df, 'ECG')
        # print(f"✓ Normalized ECG signals")
        
        return df
    
    def save_data(self, df: pd.DataFrame) -> None:
        """
        Save the preprocessed dataframe to a pickle file.
        
        Args:
            df: Preprocessed dataframe to save
        """
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        df.to_pickle(self.output_path)
        print(f"\n✓ Saved preprocessed dataframe with shape {df.shape} to {self.output_path}")
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline:
        1. Load data
        2. Preprocess data
        3. Save processed data
        
        Returns:
            Preprocessed dataframe
        """
        print("=" * 80)
        print("PPG/ECG Preprocessing Pipeline")
        print("=" * 80)
        print(f"Input file: {self.input_path}")
        print(f"Output file: {self.output_path}")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"Bandpass filter: {self.bandpass_low}-{self.bandpass_high} Hz")
        print(f"Bandpass order: {self.bandpass_order}")
        print(f"IQR clipping: k={self.iqr_k}")
        print("=" * 80)
        
        # Load data
        df = load_pd_file(self.input_path)

        
        # Preprocess data
        df_processed = self.preprocess_dataframe(df)
        
        # Save processed data
        self.save_data(df_processed)
        
        print("\n" + "=" * 80)
        print("Preprocessing complete!")
        print("=" * 80)
        
        return df_processed


