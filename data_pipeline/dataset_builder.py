import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.io
from biomarkers.extract_biomarkers import BiomarkerExtractor
import matplotlib.pyplot as plt
from utils import delete_empty_dirs, load_mat_file, extract_ppg_ecg
import shutil

class CombinedDataBuilder:
    """Build a combined dataframe using BiomarkerExtractor to compute sqi, hr, fp, and bm.

    The class writes temporary segment files into `segment_save_path` and uses
    `BiomarkerExtractor` to derive SQI, HR, fiducials and biomarkers for each segment.
    """
    def __init__(self,
                 fs,
                 mat_file_path: str,
                 csv_file_path: str,
                 segment_save_path: str,
                 fs_default: int = 240,
                 pad_width: int = 960,
                 tile_reps: int = 1,
                 start_sig: int = 0,
                 end_sig: int = -1,
                 use_tk: bool = False,
                 output_dir: str = None,
                 output_filename: str = None):
        self.fs = fs
        self.mat_file_path = mat_file_path
        self.csv_file_path = csv_file_path
        self.segment_save_path = segment_save_path
        self.fs_default = fs_default
        self.pad_width = pad_width
        self.tile_reps = tile_reps
        self.start_sig = start_sig
        self.end_sig = end_sig
        self.use_tk = use_tk
        self.output_dir = output_dir
        self.output_filename = output_filename

    def build(self) -> pd.DataFrame:
        mat_data = load_mat_file(self.mat_file_path)
        ppg, ecg = extract_ppg_ecg(mat_data)

        metadata_df = pd.read_csv(self.csv_file_path)

        required_columns = {"ECGcat", "Filename", "ID0"}
        missing = required_columns - set(metadata_df.columns)
        if missing:
            raise KeyError(f"Missing required columns in CSV: {', '.join(sorted(missing))}")

        num_samples = len(metadata_df)
        if ppg.shape[0] != num_samples or ecg.shape[0] != num_samples:
            raise ValueError("Mismatch between metadata rows and signal samples. Ensure the .mat and .csv files are aligned.")

        # Ensure directories exist for mat/temp_mat
        os.makedirs(os.path.join(self.segment_save_path, "mat"), exist_ok=True)
        os.makedirs(os.path.join(self.segment_save_path, "temp_mat"), exist_ok=True)

        # Instantiate extractor; mat_save_path and temp_mat_save_path are relative names
        extractor = BiomarkerExtractor(
                                       fs=self.fs,
                                       mat_save_path="mat",
                                       temp_mat_save_path="temp_mat",
                                       savingfolder=self.segment_save_path,
                                       start_sig=self.start_sig,
                                       end_sig=self.end_sig,
                                       pad_width=self.pad_width,
                                       tile_reps=self.tile_reps,
                                       use_tk=self.use_tk)

        rows = []
        for i in range(num_samples):
            ppg_sig = np.asarray(ppg[i])
            ecg_sig = np.asarray(ecg[i])
            label = metadata_df.loc[i, "ECGcat"]
            fname = str(metadata_df.loc[i, "Filename"])
            id0 = metadata_df.loc[i, "ID0"]

            sqi, sqi_sample, hr, fp_new, bm_stats, hrv = extractor.extract_biomarkers_from_fiducial_points(ppg_sig, signal_index=i)
            
            # Check that sqi_sample has the same length as PPG
            if len(sqi_sample) != len(ppg_sig):
                raise ValueError(f"SQI sample length {len(sqi_sample)} does not match PPG length {len(ppg_sig)} for sample {i}")
            
            bm_features = self.extract_features_from_bm_stats(bm_stats['ppg_sig'])
   
            rows.append({
                "ECG": ecg_sig,
                "PPG": ppg_sig,
                "label": label,
                "Filename": fname,
                "ID0": id0,
                "sqi": sqi,
                "sqi_sample": sqi_sample,
                "hr": hr,
                "fp": fp_new,
                "bm": bm_features,
                "hrv": hrv
            })

        combined_df = pd.DataFrame(rows)
        # Add index column
        combined_df.insert(0, "index", combined_df.index.to_numpy())

        # Cleanup any empty directories created during extraction
        delete_empty_dirs(self.segment_save_path)

        print(f"Computed biomarkers for {num_samples} segments. All segments kept (no filtering applied).")

        # Save combined dataframe using configured output path
        try:
            self.save_combined(combined_df)
        except Exception as e:
            print(f"Warning: failed to save combined dataframe: {e}")

        return combined_df
    
    def extract_features_from_bm_stats(self, bm_stats: Dict[str, Any]) -> Dict[str, Any]:
        features = {}

        for key in bm_stats.keys():
            df = bm_stats[key]
            # Flatten the DataFrame rows into feature names like Tpi_mean, Tpi_std, etc.
            for stat_name, value in df.items():
                features[f"{key}_{stat_name}"] = float(value)
        return features

    def save_combined(self, df: pd.DataFrame, output_dir: str = None, filename: str = None) -> str:
        """Save the combined dataframe to disk and return the full path.

        If `output_dir` or `filename` are not provided, use the values set on the builder.
        """
        out_dir = output_dir or self.output_dir or OUTPUT_DIR
        out_name = filename or self.output_filename or OUTPUT_FILENAME
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_name)
        df.to_pickle(out_path)
        print(f"Saved combined dataframe with shape {df.shape} to {out_path}")
        # Remove the temporary segment directory to save space (safe-guarded)
        try:
            seg_path = getattr(self, 'segment_save_path', None)
            if seg_path and os.path.isdir(seg_path):
                abs_seg = os.path.abspath(seg_path)
                # Basic safety checks: don't delete root and require reasonable path length
                if abs_seg not in (os.path.abspath('/'), '') and len(abs_seg) > 5:
                    shutil.rmtree(abs_seg)
                    print(f"Deleted temporary segment directory: {abs_seg}")
                else:
                    print(f"Skipping deletion of unsafe segment path: {abs_seg}")
        except Exception as e:
            print(f"Warning: failed to delete segment directory {seg_path}: {e}")
        return out_path

    def filter_and_save_by_thresholds(self, 
                                       df: pd.DataFrame, 
                                       thresholds: list,
                                       output_subdir: str = "combined_df") -> dict:
        """
        Filter dataframe for multiple SQI thresholds, calculate prevalence for each,
        and save clean dataframes.
        
        Args:
            df: The combined dataframe to filter
            thresholds: List of SQI threshold values to filter by
            output_subdir: Subdirectory name within output_dir to save filtered dataframes
            
        Returns:
            Dictionary with threshold as key and prevalence stats as value
        """
        results = {}
        output_path_full = os.path.join(self.output_dir, output_subdir)
        os.makedirs(output_path_full, exist_ok=True)
        
        # Calculate prevalence before filtering
        prevalence_before = calculate_ectopic_prevalence(df)
        print("\n" + "="*80)
        print("FILTERING BY MULTIPLE SQI THRESHOLDS AND CALCULATING PREVALENCE")
        print("="*80)
        print_prevalence_report(prevalence_before, None)
        
        # Filter and calculate prevalence for each threshold
        for threshold in thresholds:
            print(f"\n{'='*80}")
            print(f"Processing Threshold: {threshold}")
            print(f"{'='*80}")
            
            # Filter dataframe by SQI threshold
            keep_mask = df["sqi"] >= threshold
            clean_df = df[keep_mask].copy().reset_index(drop=True)
            
            # Update the index column with the new index values
            if "index" in clean_df.columns:
                clean_df["index"] = clean_df.index.to_numpy()
            
            num_kept = len(clean_df)
            num_removed = len(df) - num_kept
            
            if num_kept == 0:
                print(f"WARNING - Threshold {threshold}: All segments filtered out!")
                results[threshold] = {
                    'status': 'all_filtered',
                    'num_kept': 0,
                    'num_removed': num_removed,
                    'prevalence': None
                }
                continue
            
            print(f"Kept {num_kept} / {len(df)} segments ({num_kept / len(df) * 100:.2f}%)")
            print(f"Removed {num_removed} segments ({num_removed / len(df) * 100:.2f}%)")
            
            # Calculate prevalence after filtering
            prevalence_after = calculate_ectopic_prevalence(clean_df)
            print_prevalence_report(prevalence_before, prevalence_after, threshold=threshold)
            
            # Save clean dataframe for this threshold
            clean_output_path = os.path.join(
                output_path_full, 
                f"ppg_ecg_label_filename_sqi{threshold}.pkl"
            )
            clean_df.to_pickle(clean_output_path)
            print(f"\n✓ Saved clean dataframe ({clean_df.shape}) to {clean_output_path}")
            
            # Store results
            results[threshold] = {
                'status': 'success',
                'num_kept': num_kept,
                'num_removed': num_removed,
                'prevalence': prevalence_after,
                'output_path': clean_output_path
            }
        
        # Print summary
        print(f"\n{'='*80}")
        print("FILTERING SUMMARY")
        print(f"{'='*80}")
        print(f"{'Threshold':<12} {'Kept':<12} {'Removed':<12} {'Ectopic %':<15} {'Normal %':<15}")
        print("-" * 80)
        for threshold in sorted(thresholds):
            if threshold in results:
                r = results[threshold]
                if r['status'] == 'success' and r['prevalence']:
                    ectopic_pct = r['prevalence'].get('ectopic_percentage', 0)
                    normal_pct = r['prevalence'].get('normal_percentage', 0)
                    print(f"{threshold:<12} {r['num_kept']:<12} {r['num_removed']:<12} "
                          f"{ectopic_pct:<15.2f} {normal_pct:<15.2f}")
                else:
                    print(f"{threshold:<12} {'ALL FILTERED':<12} {r['num_removed']:<12} {'N/A':<15} {'N/A':<15}")
        
        # Save results to text file
        self._save_filtering_results_to_txt(
            results=results,
            prevalence_before=prevalence_before,
            thresholds=thresholds,
            output_path_full=output_path_full
        )
        
        return results
    
    def _save_filtering_results_to_txt(self,
                                       results: dict,
                                       prevalence_before: dict,
                                       thresholds: list,
                                       output_path_full: str) -> None:
        """
        Save filtering results and prevalence statistics to a text file.
        
        Args:
            results: Dictionary with threshold as key and filtering results as value
            prevalence_before: Prevalence statistics before filtering
            thresholds: List of thresholds processed
            output_path_full: Full path to output directory
        """
        report_path = os.path.join(output_path_full, "filtering_results_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FILTERING RESULTS AND PREVALENCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics before filtering
            f.write("OVERALL STATISTICS (BEFORE FILTERING)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Segments: {prevalence_before['total_segments']}\n\n")
            
            f.write("Label Distribution:\n")
            for label, count in prevalence_before['label_counts'].items():
                pct = prevalence_before['label_percentages'].get(label, 0)
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")
            
            f.write(f"\nEctopic Prevalence: {prevalence_before['ectopic_count']} "
                   f"({prevalence_before['ectopic_percentage']:.2f}%)\n")
            f.write(f"Normal Prevalence: {prevalence_before['normal_count']} "
                   f"({prevalence_before['normal_percentage']:.2f}%)\n")
            
            # Results for each threshold
            f.write("\n" + "=" * 80 + "\n")
            f.write("FILTERING RESULTS BY THRESHOLD\n")
            f.write("=" * 80 + "\n\n")
            
            for threshold in sorted(thresholds):
                if threshold not in results:
                    continue
                
                r = results[threshold]
                f.write(f"Threshold: {threshold}\n")
                f.write("-" * 80 + "\n")
                
                if r['status'] == 'all_filtered':
                    f.write(f"Status: ALL SEGMENTS FILTERED OUT\n")
                    f.write(f"Segments Removed: {r['num_removed']}\n")
                    f.write(f"Segments Kept: 0\n\n")
                    continue
                
                f.write(f"Status: SUCCESS\n")
                f.write(f"Segments Kept: {r['num_kept']}\n")
                f.write(f"Segments Removed: {r['num_removed']}\n")
                f.write(f"Retention Rate: {r['num_kept'] / (r['num_kept'] + r['num_removed']) * 100:.2f}%\n")
                f.write(f"Output Path: {r.get('output_path', 'N/A')}\n\n")
                
                if r['prevalence']:
                    prev = r['prevalence']
                    f.write("Prevalence After Filtering:\n")
                    f.write(f"  Total Segments: {prev['total_segments']}\n")
                    
                    f.write("  Label Distribution:\n")
                    for label, count in prev['label_counts'].items():
                        pct = prev['label_percentages'].get(label, 0)
                        f.write(f"    {label}: {count} ({pct:.2f}%)\n")
                    
                    f.write(f"  Ectopic Prevalence: {prev['ectopic_count']} "
                           f"({prev['ectopic_percentage']:.2f}%)\n")
                    f.write(f"  Normal Prevalence: {prev['normal_count']} "
                           f"({prev['normal_percentage']:.2f}%)\n")
                    
                    # Calculate change from before
                    ectopic_change = prev['ectopic_percentage'] - prevalence_before['ectopic_percentage']
                    normal_change = prev['normal_percentage'] - prevalence_before['normal_percentage']
                    f.write(f"\n  Change from Before Filtering:\n")
                    f.write(f"    Ectopic: {ectopic_change:+.2f}%\n")
                    f.write(f"    Normal: {normal_change:+.2f}%\n")
                
                f.write("\n")
            
            # Summary table
            f.write("=" * 80 + "\n")
            f.write("SUMMARY TABLE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Threshold':<12} {'Kept':<12} {'Removed':<12} {'Retention %':<15} "
                   f"{'Ectopic %':<15} {'Normal %':<15}\n")
            f.write("-" * 80 + "\n")
            
            for threshold in sorted(thresholds):
                if threshold not in results:
                    continue
                
                r = results[threshold]
                if r['status'] == 'success' and r['prevalence']:
                    retention_pct = (r['num_kept'] / (r['num_kept'] + r['num_removed']) * 100) if (r['num_kept'] + r['num_removed']) > 0 else 0
                    ectopic_pct = r['prevalence'].get('ectopic_percentage', 0)
                    normal_pct = r['prevalence'].get('normal_percentage', 0)
                    f.write(f"{threshold:<12} {r['num_kept']:<12} {r['num_removed']:<12} "
                           f"{retention_pct:<15.2f} {ectopic_pct:<15.2f} {normal_pct:<15.2f}\n")
                else:
                    f.write(f"{threshold:<12} {'ALL FILTERED':<12} {r['num_removed']:<12} "
                           f"{'0.00':<15} {'N/A':<15} {'N/A':<15}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Saved filtering results report to {report_path}")
    
    def plot_sqi_histogram(self, df: pd.DataFrame, threshold: float) -> None:
        """
        Plot a histogram of all segment SQI values with a vertical line at the specified threshold.
        """
        histogram_path = os.path.join(self.output_dir, "sqi_histogram.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter to only include NORM, PAC, and PVC labels
        keep_mask = df["label"].isin(['NORM', 'PAC', 'PVC', 'ECT'])
        sqi_values = df[keep_mask]["sqi"].values
        ax.hist(sqi_values, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
        
        ax.set_xlabel('SQI Score', fontsize=12)
        ax.set_ylabel('Number of Segments', fontsize=12)
        ax.set_title('Distribution of SQI Scores for All Segments', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_sqi = np.mean(sqi_values)
        median_sqi = np.median(sqi_values)
        above_threshold = np.sum(sqi_values >= threshold)
        below_threshold = len(sqi_values) - above_threshold
        
        stats_text = f'Mean: {mean_sqi:.2f}\nMedian: {median_sqi:.2f}\n'
        stats_text += f'Above threshold ({threshold}): {above_threshold} ({above_threshold/len(sqi_values)*100:.1f}%)\n'
        stats_text += f'Below threshold ({threshold}): {below_threshold} ({below_threshold/len(sqi_values)*100:.1f}%)'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150)
        plt.close(fig)
        print(f"Saved SQI histogram to {histogram_path}")


    def plot_noisy_examples(self,
                            df: pd.DataFrame,
                            threshold: float,
                            output_subdir: str = "noisy_examples",
                            max_examples: int = 20,
                            threshold_subdir: bool = True) -> int:
        """
        Plot noisy examples (segments below SQI threshold) as a method of CombinedDataBuilder.
        
        Args:
            df: The combined dataframe containing PPG, ECG, and SQI data
            threshold: SQI threshold value - segments below this will be considered noisy
            output_subdir: Subdirectory name within output_dir to save noisy example plots
            max_examples: Maximum number of noisy examples to plot
            threshold_subdir: If True, create a subdirectory named by threshold (e.g., "noisy_examples_sqi50")
            
        Returns:
            Number of plots saved
        """
        # Filter noisy examples (below threshold)
        noisy_mask = df["sqi"] < threshold
        noisy_df = df[noisy_mask].copy().reset_index(drop=True)
        
        if noisy_df.empty:
            print(f"No noisy examples found below threshold {threshold}")
            return 0
        
        # Determine output directory
        if threshold_subdir:
            output_dir = os.path.join(self.output_dir, output_subdir, f"sqi{int(threshold)}")
        else:
            output_dir = os.path.join(self.output_dir, output_subdir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Randomly sample noisy examples
        n = min(max_examples, len(noisy_df))
        sampled = noisy_df.sample(n=n, random_state=42)
        
        saved = 0
        print(f"\nPlotting {n} noisy examples (SQI < {threshold}) to {output_dir}...")
        
        for idx, row in sampled.iterrows():
            ppg_signal = np.asarray(row["PPG"]).squeeze()
            ecg_signal = np.asarray(row["ECG"]).squeeze()
            fname = row.get("Filename", f"sample_{idx}")
            sqi = row.get("sqi", np.nan)
            label = row.get("label", "Unknown")
            id0 = row.get("ID0", "Unknown")
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            
            # Plot PPG signal
            axes[0].plot(ppg_signal, color="tab:blue", linewidth=1)
            axes[0].set_ylabel("PPG", fontsize=11)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f"Patient ID: {id0} | Label: {label} | SQI: {sqi:.2f}", 
                            fontsize=10, pad=10)
            
            # Plot ECG signal
            axes[1].plot(ecg_signal, color="tab:red", linewidth=1)
            axes[1].set_ylabel("ECG", fontsize=11)
            axes[1].set_xlabel("Samples", fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            fig.suptitle(f"Noisy Example | File: {fname}", fontsize=12, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            safe_name = self._sanitize_filename(str(fname))
            out_path = os.path.join(output_dir, f"noisy_{saved+1:02d}_{safe_name}.png")
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved += 1
        
        print(f"✓ Saved {saved} noisy example plots to {output_dir}")
        print(f"  Total noisy segments below threshold {threshold}: {len(noisy_df)}")
        
        return saved

    def run_pipeline(self, config) -> pd.DataFrame:
        """
        Run the complete dataset building pipeline:
        1. Build combined dataset
        2. Filter by multiple thresholds (if config provided)
        3. Plot noisy examples (if config provided)
        
        Args:
            config: Full config object containing extractor, filtering, and noisy_examples sections
        
        Returns:
            Combined dataframe
        """
        print("=" * 80)
        print("DATASET BUILDING PIPELINE")
        print("=" * 80)
        
        # Step 1: Build combined dataset
        print("\n" + "=" * 80)
        print("STEP 1: BUILDING COMBINED DATASET")
        print("=" * 80)
        combined_df = self.build()
        
        # Step 2: Filter by multiple thresholds if filtering config is provided
        if hasattr(config, 'filtering') and config.filtering:
            thresholds = config.filtering.get('sqi_thresholds', [])
            output_subdir = config.filtering.get('output_subdir', 'combined_df')
            
            if thresholds:
                print("\n" + "=" * 80)
                print("STEP 2: FILTERING BY MULTIPLE SQI THRESHOLDS")
                print("=" * 80)
                results = self.filter_and_save_by_thresholds(
                    df=combined_df,
                    thresholds=thresholds,
                    output_subdir=output_subdir
                )
                print(f"\n✓ Completed filtering for {len(thresholds)} thresholds")
            else:
                print("\nWarning: No thresholds specified in filtering.sqi_thresholds")
        
        # Step 3: Plot noisy examples if noisy_examples config is provided
        if hasattr(config, 'noisy_examples') and config.noisy_examples:
            noisy_cfg = config.noisy_examples
            if noisy_cfg.get('enable', False):
                threshold = noisy_cfg.get('threshold', 50)
                output_subdir = noisy_cfg.get('output_subdir', 'noisy_examples')
                max_examples = noisy_cfg.get('max_examples', 20)
                threshold_subdir = noisy_cfg.get('threshold_subdir', True)
                
                print("\n" + "=" * 80)
                print("STEP 3: PLOTTING NOISY EXAMPLES")
                print("=" * 80)
                num_plotted = self.plot_noisy_examples(
                    df=combined_df,
                    threshold=threshold,
                    output_subdir=output_subdir,
                    max_examples=max_examples,
                    threshold_subdir=threshold_subdir
                )
                if num_plotted > 0:
                    print(f"\n✓ Completed plotting {num_plotted} noisy examples")
                else:
                    print(f"\nNo noisy examples found below threshold {threshold}")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        
        return combined_df

    def _sanitize_filename(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_", ".", " ") else "_" for c in str(name))
        return safe[:200] if len(safe) > 200 else safe
 
 
def calculate_ectopic_prevalence(df: pd.DataFrame) -> dict:
    """
    Calculate the prevalence of ectopics in the dataset.
    Returns a dictionary with counts and percentages for each label.
    """
    total = len(df)
    label_counts = df['label'].value_counts()
    label_percentages = df['label'].value_counts(normalize=True) * 100
    
    result = {
        'total_segments': total,
        'label_counts': label_counts.to_dict(),
        'label_percentages': label_percentages.to_dict()
    }
    
    # Calculate ectopic prevalence (non-NORM labels)
    ectopic_labels = [label for label in label_counts.index if str(label) == 'PAC' or str(label) == 'PVC' or str(label) == 'ECT']
    ectopic_count = sum(label_counts[label] for label in ectopic_labels)
    normal_count = label_counts.get('NORM', 0)
    base = ectopic_count + normal_count
    ectopic_percentage = (ectopic_count / base * 100) if  base > 0 else 0
    
    result['ectopic_count'] = ectopic_count
    result['ectopic_percentage'] = ectopic_percentage
    result['normal_count'] = normal_count
    result['normal_percentage'] = (result['normal_count'] / base * 100) if base > 0 else 0
    
    return result


def print_prevalence_report(before: dict, after: dict, threshold: int = None) -> None:
    """
    Print a formatted report comparing prevalence before and after filtering.
    """
    if threshold is not None:
        print(f"\n{'='*60}")
        print(f"Prevalence Report - Threshold: {threshold}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Prevalence Report - Before Filtering")
        print(f"{'='*60}")
    
    print(f"\nTotal Segments:")
    print(f"  Before: {before['total_segments']}")
    if after:
        print(f"  After:  {after['total_segments']} (Removed: {before['total_segments'] - after['total_segments']})")
    
    print(f"\nEctopic Prevalence:")
    print(f"  Before: {before['ectopic_count']} ({before['ectopic_percentage']:.2f}%)")
    if after:
        print(f"  After:  {after['ectopic_count']} ({after['ectopic_percentage']:.2f}%)")
        change = after['ectopic_percentage'] - before['ectopic_percentage']
        print(f"  Change: {change:+.2f}%")
    
    print(f"\nNormal Prevalence:")
    print(f"  Before: {before['normal_count']} ({before['normal_percentage']:.2f}%)")
    if after:
        print(f"  After:  {after['normal_count']} ({after['normal_percentage']:.2f}%)")
        change = after['normal_percentage'] - before['normal_percentage']
        print(f"  Change: {change:+.2f}%")
    
    print(f"\nLabel Distribution:")
    for label in sorted(set(list(before['label_counts'].keys()) + (list(after['label_counts'].keys()) if after else []))):
        before_count = before['label_counts'].get(label, 0)
        before_pct = before['label_percentages'].get(label, 0)
        if after:
            after_count = after['label_counts'].get(label, 0)
            after_pct = after['label_percentages'].get(label, 0)
            print(f"  {label}: {before_count} ({before_pct:.2f}%) -> {after_count} ({after_pct:.2f}%)")
        else:
            print(f"  {label}: {before_count} ({before_pct:.2f}%)")




