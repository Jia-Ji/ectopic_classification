import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from omegaconf import DictConfig

class DataSplitter:
    """
    Data splitting class for splitting datasets by patient ID.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize DataSplitter with configuration from config file.
        
        Args:
            config: DictConfig containing splitting parameters
        """
        self.config = config
        self.split_cfg = config.split_data
        
        # Load parameters from config
        self.input_path = self.split_cfg.get('input_path')
        self.output_dir = self.split_cfg.get('output_dir')
        self.target_labels = self.split_cfg.get('target_labels', ['NORM', 'ECT'])
        self.label_mapping = self.split_cfg.get('label_mapping', {'NORM': 0, 'ECT': 1})
        self.train_prop = self.split_cfg.get('train_prop', 0.6)
        self.val_prop = self.split_cfg.get('val_prop', 0.2)
        self.test_prop = self.split_cfg.get('test_prop', 0.2)
        
        # Validate required paths
        if not self.input_path:
            raise ValueError("input_path must be specified in config.split_data")
        if not self.output_dir:
            raise ValueError("output_dir must be specified in config.split_data")
        
        # Validate proportions sum to 1
        if abs(self.train_prop + self.val_prop + self.test_prop - 1.0) > 1e-6:
            raise ValueError(f"Split proportions must sum to 1.0, got {self.train_prop + self.val_prop + self.test_prop}")
        
        # Initialize random_state
        self.random_state = None
    
    def load_filtered_data(self) -> pd.DataFrame:
        """Load the pickle file from preprocessing"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        df = pd.read_pickle(self.input_path)
        print(f"Loaded dataframe with shape {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def filter_and_encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to only include target labels and encode them.
        Merges 'PAC' and 'PVC' into 'ECT' class.
        Keeps 'VT' as a separate class.
        """
        # Check for case variations (Norm/NORM, PAC, PVC)
        available_labels = df['label'].unique()
        print(f"\nAvailable labels in data: {available_labels}")
        
        # Define source labels that will be merged into target labels
        # NORM maps to NORM, PAC and PVC map to ECT
        source_labels_to_target = {
            'NORM': 'NORM',
            'PAC': 'ECT',
            'PVC': 'ECT',
            'ECT': 'ECT',
        }
        
        # Try to match source labels (case-insensitive)
        matched_source_labels = []
        label_mapping_normalized = {}
        
        for source_label, target_label in source_labels_to_target.items():
            # Try exact match first
            if source_label in available_labels:
                matched_source_labels.append(source_label)
                label_mapping_normalized[source_label] = self.label_mapping[target_label]
            # Try uppercase
            elif source_label.upper() in available_labels:
                matched_label = source_label.upper()
                matched_source_labels.append(matched_label)
                label_mapping_normalized[matched_label] = self.label_mapping[target_label]
            # Try case-insensitive match
            else:
                for avail_label in available_labels:
                    if str(avail_label).upper() == source_label.upper():
                        matched_source_labels.append(avail_label)
                        label_mapping_normalized[avail_label] = self.label_mapping[target_label]
                        break
        
        if not matched_source_labels:
            raise ValueError(f"None of the source labels {list(source_labels_to_target.keys())} found in data. Available labels: {available_labels}")
        
        print(f"\nMatched source labels: {matched_source_labels}")
        print(f"Label mapping (source -> encoded): {label_mapping_normalized}")
        
        # Filter dataframe to only include matched source labels
        df_filtered = df[df['label'].isin(matched_source_labels)].copy()
        print(f"\nFiltered dataframe shape: {df_filtered.shape}")
        print(f"Label distribution after filtering (before merging):\n{df_filtered['label'].value_counts()}")
        
        # Map source labels to target labels (NORM stays NORM, PAC/PVC become ECT)
        def map_to_target_label(source_label):
            for src, tgt in source_labels_to_target.items():
                if str(source_label).upper() == src.upper():
                    return tgt
            return source_label
        
        df_filtered['label'] = df_filtered['label'].apply(map_to_target_label)
        print(f"\nLabel distribution after merging to target labels:\n{df_filtered['label'].value_counts()}")
        
        # Encode labels using target label mapping
        df_filtered['encoded_label'] = df_filtered['label'].map(self.label_mapping)
        
        # Check for any unmapped labels
        if df_filtered['encoded_label'].isna().any():
            unmapped = df_filtered[df_filtered['encoded_label'].isna()]['label'].unique()
            raise ValueError(f"Some labels could not be mapped: {unmapped}")
        
        return df_filtered
    
    def split_by_patient_id(self, df: pd.DataFrame) -> tuple:
        """
        Split dataset by patient ID (ID0) to avoid data leakage.
        Uses MultilabelStratifiedKFold with ectopic proportions per patient as labels.
        Ensures similar prevalence of 'ECT' across splits using stratified split.
        
        Returns:
            Tuple of (df_train, df_val, df_test)
        """
        # Create a summary of labels per patient
        summary = (
            df.groupby(['ID0', 'label'])
            .size()
            .reset_index(name='count')
        )
        
        # Create pivot table: rows = ID0, columns = label, values = count
        pivot = summary.pivot(index='ID0', columns='label', values='count').fillna(0)
        
        # Calculate total beats per patient
        pivot['total'] = pivot.sum(axis=1)
        
        # Calculate proportions of each label per patient
        # For multilabel stratification, we use proportions instead of counts
        # Each patient is represented by proportions: [proportion_NORM, proportion_ECT, proportion_VT]
        label_cols = [col for col in pivot.columns if col != 'total']
        proportions = pivot[label_cols].div(pivot['total'], axis=0).fillna(0)
        
        # Prepare data for MultilabelStratifiedKFold
        # X: patient IDs, y: label proportions per patient
        patient_ids = np.array(pivot.index)
        y = proportions.values
        
        print(f"\nTotal unique patients (ID0): {len(patient_ids)}")
        print(f"Label distribution per patient (summary):")
        print(pivot[label_cols].sum())
        print(f"\nEctopic proportion statistics per patient:")
        if 'ECT' in proportions.columns:
            print(f"  Mean ECT proportion: {proportions['ECT'].mean():.4f}")
            print(f"  Std ECT proportion: {proportions['ECT'].std():.4f}")
            print(f"  Min ECT proportion: {proportions['ECT'].min():.4f}")
            print(f"  Max ECT proportion: {proportions['ECT'].max():.4f}")
        
        # Determine number of folds based on proportions
        # Use a reasonable number of folds (e.g., 5 or 10) to allow flexible splitting
        n_folds =10
        random_state = 103
        kfold = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Store random_state as instance variable for reporting
        self.random_state = random_state
        
        # Get all fold splits
        # Each fold returns (train_indices, test_indices)
        fold_splits = list(kfold.split(patient_ids, y))
        
        # Calculate how many test folds to assign to val and test based on proportions
        # In KFold, each fold's test set contains 1/n_folds of the data
        # We'll assign some folds' test sets to val, some to test, and the rest to train
        val_test_total = self.val_prop + self.test_prop
        n_test_folds = round(n_folds * val_test_total)  # Number of folds to use for val+test
        n_val_folds = round(n_test_folds * (self.val_prop / val_test_total)) if val_test_total > 0 else 0
        n_test_folds_actual = n_test_folds - n_val_folds
        
        # Ensure we have at least one fold for each split
        if n_test_folds == 0:
            n_test_folds = 1
            n_val_folds = 0
            n_test_folds_actual = 1
        if n_val_folds == 0 and self.val_prop > 0:
            n_val_folds = 1
            n_test_folds_actual = max(1, n_test_folds_actual - 1)
        
        print(f"\nUsing {n_folds}-fold split:")
        print(f"  Val test folds: {n_val_folds}, Test folds: {n_test_folds_actual}, Train: remaining")
        
        # Collect indices from fold test sets
        # In KFold, each patient appears in exactly one test fold
        val_indices = []
        test_indices = []
        
        # Assign first n_val_folds' test sets to validation
        for i in range(n_val_folds):
            val_indices.extend(fold_splits[i][1])  # Test indices from fold i
        
        # Assign next n_test_folds_actual' test sets to test
        for i in range(n_val_folds, n_val_folds + n_test_folds_actual):
            test_indices.extend(fold_splits[i][1])  # Test indices from fold i
        
        # All remaining patients go to train
        val_set_indices = set(val_indices)
        test_set_indices = set(test_indices)
        train_indices = [i for i in range(len(patient_ids)) if i not in val_set_indices and i not in test_set_indices]
        
        # Get unique patient IDs
        train_patients = patient_ids[train_indices]
        val_patients = patient_ids[val_indices]
        test_patients = patient_ids[test_indices]
        
        print(f"\nSplit sizes:")
        print(f"Train patients: {len(train_patients)}")
        print(f"Val patients: {len(val_patients)}")
        print(f"Test patients: {len(test_patients)}")
        
        # Verify no overlap
        train_set = set(train_patients)
        val_set = set(val_patients)
        test_set = set(test_patients)
        
        assert len(train_set & val_set) == 0, "Overlap between train and val patients!"
        assert len(train_set & test_set) == 0, "Overlap between train and test patients!"
        assert len(val_set & test_set) == 0, "Overlap between val and test patients!"
        print("\n✓ No patient overlap between splits")
        
        # Calculate and print label prevalence in each split
        print(f"\nLabel prevalence per split:")
        for label in self.target_labels:
            if label in proportions.columns:
                train_prop = proportions.loc[train_patients, label].mean()
                val_prop = proportions.loc[val_patients, label].mean()
                test_prop = proportions.loc[test_patients, label].mean()
                print(f"  {label} - Train: {train_prop:.4f}, Val: {val_prop:.4f}, Test: {test_prop:.4f}")
                max_diff = max(abs(train_prop - val_prop), abs(train_prop - test_prop), abs(val_prop - test_prop))
                print(f"    Max difference: {max_diff:.4f}")
        
        # Create splits
        df_train = df[df['ID0'].isin(train_patients)].copy()
        df_val = df[df['ID0'].isin(val_patients)].copy()
        df_test = df[df['ID0'].isin(test_patients)].copy()
        
        return df_train, df_val, df_test
    
    def fp_to_mask(self, fp: pd.DataFrame, signal_length: int) -> np.ndarray:
        """
        Convert fiducial points DataFrame to a 4-channel mask.
        
        Args:
            fp: DataFrame with columns 'on', 'sp', 'dn', 'dp' containing fiducial point indices.
                Each column contains a Series/array/list of indices where that fiducial point occurs.
            signal_length: Length of the signal to create mask for
            
        Returns:
            4-channel mask array of shape (signal_length, 4) where each channel corresponds to:
            channel 0: 'on' (pulse onset)
            channel 1: 'sp' (systolic peak)
            channel 2: 'dn' (dicrotic notch)
            channel 3: 'dp' (diastolic peak)
        """
        mask = np.zeros((signal_length, 4), dtype=np.float32)
        
        # Extract fiducial point columns
        fp_columns = ['on', 'sp', 'dn', 'dp']
        channel_idx = 0
        
        for col in fp_columns:
            if col in fp.columns:
                fp_values = fp[col]

                fp_values = fp_values.to_numpy()
                assert isinstance(fp_values, (list, np.ndarray)), \
                f"fp_values must be a list or numpy array, but got {type(fp_values)}, mask will be all zeros.."

                # Convert to array and filter out NaN values
                if isinstance(fp_values, (list, np.ndarray)):
                    fp_indices = np.asarray(fp_values, dtype=float)
                    # Filter out NaN values
                    fp_indices = fp_indices[~np.isnan(fp_indices)]
                    # Convert to integers
                    fp_indices = fp_indices.astype(int)
                    # Ensure indices are within signal bounds
                    fp_indices = fp_indices[(fp_indices >= 0) & (fp_indices < signal_length)]
                    # Set mask to 1 at fiducial point locations
                    if len(fp_indices) > 0:
                        mask[fp_indices, channel_idx] = 1.0
            channel_idx += 1
        
        return mask
    
    def extract_arrays(self, df_split: pd.DataFrame) -> tuple:
        """Extract ECG, PPG, label, filename, fp mask, hr, rmssd, sdsd, bm_stats, ID0, and sqi arrays from dataframe split"""
        # Ensure dataframe is properly indexed
        df_split = df_split.reset_index(drop=True)
        
        # Convert list columns to numpy arrays
        ecg_list = []
        ppg_list = []
        labels_list = []
        filenames_list = []
        fp_masks_list = []
        hr_list = []
        rmssd_list = []
        sdsd_List = []
        bm_stats_list = []
        id0_list = []
        sqi_list = []
        sqi_sample_list = []
    
        
        for idx in range(len(df_split)):
            ecg_val = df_split.iloc[idx]['ECG']
            ppg_val = df_split.iloc[idx]['PPG']
            fp_value = df_split.iloc[idx]['fp']
            label_val = df_split.iloc[idx]['encoded_label']
            filename_val = df_split.iloc[idx]['Filename']
            hr_val = df_split.iloc[idx]['hr']
            rmssd_val = df_split.iloc[idx]['hrv'].get('RMSSD', None)
            sdsd_val = df_split.iloc[idx]['hrv'].get('SDSD', None)
            bm_stats_val = df_split.iloc[idx]['bm']
            id0_val = df_split.iloc[idx]['ID0']
            sqi_val = df_split.iloc[idx]['sqi']
            sqi_sample_val = df_split.iloc[idx]['sqi_sample']

            ecg_list.append(np.asarray(ecg_val))
            ppg_array = np.asarray(ppg_val)
            ppg_list.append(np.expand_dims(ppg_array, axis=0))
            
            # Convert fp to 4-channel mask
            # fp_value should be a DataFrame with columns 'on', 'sp', 'dn', 'dp'
            signal_length = len(ppg_array)
            fp_mask = self.fp_to_mask(fp_value, signal_length) 
            
            fp_masks_list.append(fp_mask)
            labels_list.append(label_val)
            filenames_list.append(str(filename_val))
            hr_list.append(hr_val)
            rmssd_list.append(rmssd_val)
            sdsd_List.append(sdsd_val)
            bm_stats_list.append(bm_stats_val)
            id0_list.append(str(id0_val))
            sqi_list.append(float(sqi_val))
            sqi_sample_list.append(sqi_sample_val)
        
        ecg = np.array(ecg_list)
        ppg = np.array(ppg_list)
        labels = np.array(labels_list, dtype=np.int64)
        filenames = np.array(filenames_list, dtype=object)
        fp_masks = np.array(fp_masks_list)
        hr = np.array(hr_list, dtype=np.float32)
        rmssd = np.array(rmssd_list, dtype=np.float32)
        sdsd = np.array(sdsd_List, dtype=np.float32)
        bm_stats = np.array(bm_stats_list, dtype=object)
        id0 = np.array(id0_list, dtype=object)
        sqi = np.array(sqi_list, dtype=np.float32)
        sqi_sample = np.array(sqi_sample_list)
        
        # Validate all arrays have the same length
        lengths = {
            'ECG': len(ecg),
            'PPG': len(ppg),
            'labels': len(labels),
            'filenames': len(filenames),
            'fp_masks': len(fp_masks),
            'hr': len(hr),
            'rmssd': len(rmssd),
            'sdsd': len(sdsd),
            'bm_stats': len(bm_stats),
            'ID0': len(id0),
            'sqi': len(sqi),
            'sqi_sample': len(sqi_sample)
        }
        
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Length mismatch in extracted arrays: {lengths}")
        
        return ecg, ppg, labels, filenames, fp_masks, hr, rmssd, sdsd, bm_stats, id0, sqi, sqi_sample
    
    def compute_label_distribution(self, df_split: pd.DataFrame) -> tuple:
        """
        Compute label distribution for a split.
        Returns both counts and percentages as dictionaries.
        """
        # Get counts
        counts = df_split['label'].value_counts()
        
        # Get percentages
        percentages = df_split['label'].value_counts(normalize=True) * 100
        
        # Reindex to match target label order
        target_order = list(self.label_mapping.keys())
        
        # Create ordered dictionaries
        counts_dict = {}
        percentages_dict = {}
        
        for label in target_order:
            if label in counts.index:
                counts_dict[label] = int(counts[label])
                percentages_dict[label] = float(percentages[label])
            else:
                counts_dict[label] = 0
                percentages_dict[label] = 0.0
        
        return counts_dict, percentages_dict
    
    def save_splits(self, ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train,
                    ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val,
                    ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test):
        """Save train/val/test splits to output directory"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate all arrays have matching lengths before saving
        def validate_split(name, ecg, ppg, labels, filenames, fp_masks, hr, rmssd, sdsd, bm_stats, id0, sqi, sqi_sample):
            lengths = {
                'ECG': len(ecg),
                'PPG': len(ppg),
                'labels': len(labels),
                'filenames': len(filenames),
                'fp_masks': len(fp_masks),
                'hr': len(hr),
                'rmssd': len(rmssd),
                'sdsd': len(sdsd),
                'bm_stats': len(bm_stats),
                'ID0': len(id0),
                'sqi': len(sqi),
                'sqi_sample': len(sqi_sample)
            }
            if len(set(lengths.values())) > 1:
                raise ValueError(f"Length mismatch in {name} split: {lengths}")
            print(f"  {name}: All arrays have length {lengths['ECG']}")
            return lengths['ECG']
        
        print("\nValidating splits before saving...")
        train_len = validate_split("Train", ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train)
        val_len = validate_split("Val", ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val)
        test_len = validate_split("Test", ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test)
        
        # Save train split
        np.save(os.path.join(self.output_dir, 'ecg_train.npy'), ecg_train)
        np.save(os.path.join(self.output_dir, 'ppg_train.npy'), ppg_train)
        np.save(os.path.join(self.output_dir, 'labels_train.npy'), labels_train)
        np.save(os.path.join(self.output_dir, 'filenames_train.npy'), filenames_train)
        np.save(os.path.join(self.output_dir, 'fp_masks_train.npy'), fp_masks_train)
        np.save(os.path.join(self.output_dir, 'hr_train.npy'), hr_train)
        np.save(os.path.join(self.output_dir, 'rmssd_train.npy'), rmssd_train)
        np.save(os.path.join(self.output_dir, 'sdsd_train.npy'), sdsd_train)
        np.save(os.path.join(self.output_dir, 'id0_train.npy'), id0_train)
        np.save(os.path.join(self.output_dir, 'sqi_train.npy'), sqi_train)
        np.save(os.path.join(self.output_dir, 'sqi_sample_train.npy'), sqi_sample_train)
        # Save bm_stats as CSV (list of dictionaries)
        if len(bm_stats_train) > 0:
            bm_stats_df_train = pd.DataFrame(bm_stats_train)
            bm_stats_df_train.to_csv(os.path.join(self.output_dir, 'bm_stats_train.csv'), index=False)
        
        # Save val split
        np.save(os.path.join(self.output_dir, 'ecg_val.npy'), ecg_val)
        np.save(os.path.join(self.output_dir, 'ppg_val.npy'), ppg_val)
        np.save(os.path.join(self.output_dir, 'labels_val.npy'), labels_val)
        np.save(os.path.join(self.output_dir, 'filenames_val.npy'), filenames_val)
        np.save(os.path.join(self.output_dir, 'fp_masks_val.npy'), fp_masks_val)
        np.save(os.path.join(self.output_dir, 'hr_val.npy'), hr_val)
        np.save(os.path.join(self.output_dir, 'rmssd_val.npy'), rmssd_val)
        np.save(os.path.join(self.output_dir, 'sdsd_val.npy'), sdsd_val)
        np.save(os.path.join(self.output_dir, 'id0_val.npy'), id0_val)
        np.save(os.path.join(self.output_dir, 'sqi_val.npy'), sqi_val)
        np.save(os.path.join(self.output_dir, 'sqi_sample_val.npy'), sqi_sample_val)
        # Save bm_stats as CSV (list of dictionaries)
        if len(bm_stats_val) > 0:
            bm_stats_df_val = pd.DataFrame(bm_stats_val)
            bm_stats_df_val.to_csv(os.path.join(self.output_dir, 'bm_stats_val.csv'), index=False)
        
        # Save test split
        np.save(os.path.join(self.output_dir, 'ecg_test.npy'), ecg_test)
        np.save(os.path.join(self.output_dir, 'ppg_test.npy'), ppg_test)
        np.save(os.path.join(self.output_dir, 'labels_test.npy'), labels_test)
        np.save(os.path.join(self.output_dir, 'filenames_test.npy'), filenames_test)
        np.save(os.path.join(self.output_dir, 'fp_masks_test.npy'), fp_masks_test)
        np.save(os.path.join(self.output_dir, 'hr_test.npy'), hr_test)
        np.save(os.path.join(self.output_dir, 'rmssd_test.npy'), rmssd_test)
        np.save(os.path.join(self.output_dir, 'sdsd_test.npy'), sdsd_test)
        np.save(os.path.join(self.output_dir, 'id0_test.npy'), id0_test)
        np.save(os.path.join(self.output_dir, 'sqi_test.npy'), sqi_test)
        np.save(os.path.join(self.output_dir, 'sqi_sample_test.npy'), sqi_sample_test)
        # Save bm_stats as CSV (list of dictionaries)
        if len(bm_stats_test) > 0:
            bm_stats_df_test = pd.DataFrame(bm_stats_test)
            bm_stats_df_test.to_csv(os.path.join(self.output_dir, 'bm_stats_test.csv'), index=False)
        
        print(f"\n✓ Saved all splits to {self.output_dir}")
        print(f"  Train: {train_len} samples, Val: {val_len} samples, Test: {test_len} samples")
    
    def save_distribution_report(self, train_counts, train_percentages, 
                                val_counts, val_percentages,
                                test_counts, test_percentages):
        """Save label distribution report to a text file"""
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, 'data_distribution.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LABEL DISTRIBUTION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Write random seed used for splitting
            random_state = getattr(self, 'random_state', None)
            if random_state is not None:
                f.write("Random Seed (random_state):\n")
                f.write(f"  {random_state}\n")
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Write label mapping
            f.write("Label Encoding:\n")
            for label, code in self.label_mapping.items():
                f.write(f"  {label}: {code}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Train set distribution
            f.write("TRAIN SET DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total_train = sum(train_counts.values())
            f.write(f"Total samples: {total_train}\n\n")
            f.write(f"{'Label':<15} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                count = train_counts[label]
                pct = train_percentages[label]
                f.write(f"{label:<15} {count:<15} {pct:>10.2f}%\n")
            f.write("\n")
            
            # Validation set distribution
            f.write("VALIDATION SET DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total_val = sum(val_counts.values())
            f.write(f"Total samples: {total_val}\n\n")
            f.write(f"{'Label':<15} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                count = val_counts[label]
                pct = val_percentages[label]
                f.write(f"{label:<15} {count:<15} {pct:>10.2f}%\n")
            f.write("\n")
            
            # Test set distribution
            f.write("TEST SET DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            total_test = sum(test_counts.values())
            f.write(f"Total samples: {total_test}\n\n")
            f.write(f"{'Label':<15} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                count = test_counts[label]
                pct = test_percentages[label]
                f.write(f"{label:<15} {count:<15} {pct:>10.2f}%\n")
            f.write("\n")
            
            # Summary comparison
            f.write("=" * 80 + "\n")
            f.write("DISTRIBUTION COMPARISON ACROSS SPLITS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Label':<15} {'Train %':<15} {'Val %':<15} {'Test %':<15} {'Max Diff %':<15}\n")
            f.write("-" * 80 + "\n")
            for label in self.label_mapping.keys():
                train_pct = train_percentages[label]
                val_pct = val_percentages[label]
                test_pct = test_percentages[label]
                max_diff = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
                f.write(f"{label:<15} {train_pct:>10.2f}%   {val_pct:>10.2f}%   {test_pct:>10.2f}%   {max_diff:>10.2f}%\n")
            f.write("\n")
            
            # Overall statistics
            f.write("=" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total samples across all splits: {total_train + total_val + total_test}\n")
            f.write(f"Train: {total_train} ({total_train/(total_train+total_val+total_test)*100:.2f}%)\n")
            f.write(f"Validation: {total_val} ({total_val/(total_train+total_val+total_test)*100:.2f}%)\n")
            f.write(f"Test: {total_test} ({total_test/(total_train+total_val+total_test)*100:.2f}%)\n")
        
        print(f"\n✓ Saved distribution report to {report_path}")
    
    def run(self) -> None:
        """
        Run the complete data splitting pipeline:
        1. Load data
        2. Filter and encode labels
        3. Split by patient ID
        4. Extract arrays
        5. Compute distributions
        6. Save splits and reports
        """
        print("=" * 80)
        print("DATA SPLITTING PIPELINE")
        print("=" * 80)
        print(f"Input file: {self.input_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target labels: {self.target_labels}")
        print(f"Label mapping: {self.label_mapping}")
        print(f"Split proportions: Train={self.train_prop}, Val={self.val_prop}, Test={self.test_prop}")
        print("=" * 80)
        
        # Load data
        df = self.load_filtered_data()
        
        # Filter and encode labels
        df_filtered = self.filter_and_encode_labels(df)
        
        # Split by patient ID with stratified split
        df_train, df_val, df_test = self.split_by_patient_id(df_filtered)
        
        # Print dataframe info before extraction
        print(f"\n=== Dataframe Info Before Extraction ===")
        print(f"Train dataframe: {len(df_train)} rows")
        print(f"Val dataframe: {len(df_val)} rows")
        print(f"Test dataframe: {len(df_test)} rows")
        
        # Extract arrays
        print(f"\n=== Extracting Arrays ===")
        ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train = self.extract_arrays(df_train)
        print(f"✓ Extracted train arrays")
        ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val = self.extract_arrays(df_val)
        print(f"✓ Extracted val arrays")
        ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test = self.extract_arrays(df_test)
        print(f"✓ Extracted test arrays")
        
        # Print shapes
        print(f"\n=== Split Shapes ===")
        print(f"Train - ECG: {ecg_train.shape}, PPG: {ppg_train.shape}, Labels: {labels_train.shape}, Filenames: {filenames_train.shape}, FP Masks: {fp_masks_train.shape}, HR: {hr_train.shape}, rmssd: {rmssd_train.shape}, sdsd: {sdsd_train.shape}, bm_stats: {bm_stats_train.shape}, ID0: {id0_train.shape}, SQI: {sqi_train.shape}, SQI Sample: {sqi_sample_train.shape}")
        print(f"Val   - ECG: {ecg_val.shape}, PPG: {ppg_val.shape}, Labels: {labels_val.shape}, Filenames: {filenames_val.shape}, FP Masks: {fp_masks_val.shape}, HR: {hr_val.shape}, rmssd: {rmssd_val.shape}, sdsd: {sdsd_val.shape}, bm_stats: {bm_stats_val.shape}, ID0: {id0_val.shape}, SQI: {sqi_val.shape}, SQI Sample: {sqi_sample_val.shape}")
        print(f"Test  - ECG: {ecg_test.shape}, PPG: {ppg_test.shape}, Labels: {labels_test.shape}, Filenames: {filenames_test.shape}, FP Masks: {fp_masks_test.shape}, HR: {hr_test.shape}, rmssd: {rmssd_test.shape}, sdsd: {sdsd_test.shape}, bm_stats: {bm_stats_test.shape}, ID0: {id0_test.shape}, SQI: {sqi_test.shape}, SQI Sample: {sqi_sample_test.shape}")
        
        # Compute label distributions
        print(f"\n=== Label Distribution ===")
        train_counts, train_percentages = self.compute_label_distribution(df_train)
        val_counts, val_percentages = self.compute_label_distribution(df_val)
        test_counts, test_percentages = self.compute_label_distribution(df_test)
        
        # Print distributions
        print(f"\nTrain Set Distribution:")
        total_train = sum(train_counts.values())
        print(f"  Total samples: {total_train}")
        for label in self.label_mapping.keys():
            count = train_counts[label]
            pct = train_percentages[label]
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        print(f"\nValidation Set Distribution:")
        total_val = sum(val_counts.values())
        print(f"  Total samples: {total_val}")
        for label in self.label_mapping.keys():
            count = val_counts[label]
            pct = val_percentages[label]
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        print(f"\nTest Set Distribution:")
        total_test = sum(test_counts.values())
        print(f"  Total samples: {total_test}")
        for label in self.label_mapping.keys():
            count = test_counts[label]
            pct = test_percentages[label]
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        # Print distribution comparison
        print(f"\n=== Distribution Comparison Across Splits ===")
        print(f"{'Label':<15} {'Train %':<15} {'Val %':<15} {'Test %':<15} {'Max Diff %':<15}")
        print("-" * 80)
        for label in self.label_mapping.keys():
            train_pct = train_percentages[label]
            val_pct = val_percentages[label]
            test_pct = test_percentages[label]
            max_diff = max(abs(train_pct - val_pct), abs(train_pct - test_pct), abs(val_pct - test_pct))
            print(f"{label:<15} {train_pct:>10.2f}%   {val_pct:>10.2f}%   {test_pct:>10.2f}%   {max_diff:>10.2f}%")
        
        # Save splits
        self.save_splits(
            ecg_train, ppg_train, labels_train, filenames_train, fp_masks_train, hr_train, rmssd_train, sdsd_train, bm_stats_train, id0_train, sqi_train, sqi_sample_train,
            ecg_val, ppg_val, labels_val, filenames_val, fp_masks_val, hr_val, rmssd_val, sdsd_val, bm_stats_val, id0_val, sqi_val, sqi_sample_val,
            ecg_test, ppg_test, labels_test, filenames_test, fp_masks_test, hr_test, rmssd_test, sdsd_test, bm_stats_test, id0_test, sqi_test, sqi_sample_test
        )
        
        # Save distribution report
        self.save_distribution_report(
            train_counts, train_percentages,
            val_counts, val_percentages,
            test_counts, test_percentages
        )
        
        print("\n✓ Data splitting complete!")
