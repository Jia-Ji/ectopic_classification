import pandas as pd
import numpy as np
import os

DATASET_PATH = "./PPG_data/combined_df/ppg_ecg_label_filename_sqi50.pkl"

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the dataset from pickle file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_pickle(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def analyze_patients_and_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze number of patients and segments per patient.
    Returns a DataFrame with patient-level statistics.
    """
    # Count segments per patient
    patient_stats = df.groupby('ID0').agg({
        'label': 'count'  # Count number of segments
    }).rename(columns={'label': 'num_segments'})
    
    return patient_stats

def calculate_ectopic_prevalence_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate prevalence of Ectopics (PAC + PVC) per patient.
    Prevalence = (PAC + PVC) / (PAC + PVC + NORM)
    """
    # Group by patient and label to count each label type
    label_counts = df.groupby(['ID0', 'label']).size().reset_index(name='count')
    
    # Pivot to get counts per label per patient
    label_pivot = label_counts.pivot(index='ID0', columns='label', values='count').fillna(0)
    
    # Normalize column names (handle case variations)
    label_pivot.columns = [str(col).upper() for col in label_pivot.columns]
    
    # Calculate ectopic prevalence
    ectopic_count = pd.Series(0, index=label_pivot.index)
    normal_count = pd.Series(0, index=label_pivot.index)
    
    if 'PAC' in label_pivot.columns:
        ectopic_count += label_pivot['PAC']
    if 'PVC' in label_pivot.columns:
        ectopic_count += label_pivot['PVC']
    if 'NORM' in label_pivot.columns:
        normal_count = label_pivot['NORM']
    
    # Calculate total relevant segments (PAC + PVC + NORM)
    total_relevant = ectopic_count + normal_count
    
    # Calculate prevalence (avoid division by zero)
    # Convert to numpy arrays to avoid index issues
    ectopic_arr = ectopic_count.values
    normal_arr = normal_count.values
    total_arr = total_relevant.values
    
    prevalence = np.where(total_arr > 0, 
                          ectopic_arr / total_arr, 
                          0.0)
    
    # Create result dataframe - use .values to ensure no index ambiguity
    result = pd.DataFrame({
        'ID0': label_pivot.index.values,
        'ectopic_prevalence': prevalence,
        'ectopic_count': ectopic_arr,
        'normal_count': normal_arr,
        'total_relevant': total_arr
    })
    
    # Add individual label counts if they exist
    if 'PAC' in label_pivot.columns:
        result['PAC_count'] = label_pivot['PAC'].values
    else:
        result['PAC_count'] = 0
    
    if 'PVC' in label_pivot.columns:
        result['PVC_count'] = label_pivot['PVC'].values
    else:
        result['PVC_count'] = 0
    
    return result

def save_text_report(df: pd.DataFrame, patient_stats: pd.DataFrame, 
                     prevalence_df: pd.DataFrame, analysis_df: pd.DataFrame,
                     output_path: str):
    """Save comprehensive analysis report to a text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset path: {DATASET_PATH}\n\n")
        
        # Label distribution
        f.write("=" * 80 + "\n")
        f.write("LABEL DISTRIBUTION (Overall)\n")
        f.write("=" * 80 + "\n")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            f.write(f"{label}: {count} ({pct:.2f}%)\n")
        f.write(f"\nTotal segments: {len(df)}\n\n")
        
        # Patient and segment analysis
        f.write("=" * 80 + "\n")
        f.write("PATIENT AND SEGMENT ANALYSIS\n")
        f.write("=" * 80 + "\n")
        num_patients = len(patient_stats)
        total_segments = patient_stats['num_segments'].sum()
        
        f.write(f"Total number of patients: {num_patients}\n")
        f.write(f"Total number of segments: {total_segments}\n")
        f.write(f"\nSegments per patient statistics:\n")
        f.write(f"  Mean: {patient_stats['num_segments'].mean():.2f}\n")
        f.write(f"  Median: {patient_stats['num_segments'].median():.2f}\n")
        f.write(f"  Min: {patient_stats['num_segments'].min()}\n")
        f.write(f"  Max: {patient_stats['num_segments'].max()}\n")
        f.write(f"  Std: {patient_stats['num_segments'].std():.2f}\n\n")
        
        # Ectopic prevalence
        f.write("=" * 80 + "\n")
        f.write("ECTOPIC PREVALENCE PER PATIENT\n")
        f.write("=" * 80 + "\n")
        f.write("Prevalence = (PAC + PVC) / (PAC + PVC + NORM)\n\n")
        
        f.write("Ectopic prevalence statistics:\n")
        f.write(f"  Mean: {prevalence_df['ectopic_prevalence'].mean():.4f} ({prevalence_df['ectopic_prevalence'].mean()*100:.2f}%)\n")
        f.write(f"  Median: {prevalence_df['ectopic_prevalence'].median():.4f} ({prevalence_df['ectopic_prevalence'].median()*100:.2f}%)\n")
        f.write(f"  Min: {prevalence_df['ectopic_prevalence'].min():.4f} ({prevalence_df['ectopic_prevalence'].min()*100:.2f}%)\n")
        f.write(f"  Max: {prevalence_df['ectopic_prevalence'].max():.4f} ({prevalence_df['ectopic_prevalence'].max()*100:.2f}%)\n")
        f.write(f"  Std: {prevalence_df['ectopic_prevalence'].std():.4f} ({prevalence_df['ectopic_prevalence'].std()*100:.2f}%)\n\n")
        
        # Patients by prevalence ranges
        f.write("Patients by ectopic prevalence range:\n")
        ranges = [
            (0.0, 0.0, "0% (No ectopics)"),
            (0.0, 0.1, "0-10%"),
            (0.1, 0.25, "10-25%"),
            (0.25, 0.5, "25-50%"),
            (0.5, 0.75, "50-75%"),
            (0.75, 1.0, "75-100%"),
            (1.0, 1.0, "100% (All ectopics)")
        ]
        
        for low, high, label in ranges:
            if low == high:
                count = len(prevalence_df[prevalence_df['ectopic_prevalence'] == low])
            else:
                count = len(prevalence_df[(prevalence_df['ectopic_prevalence'] > low) & 
                                          (prevalence_df['ectopic_prevalence'] <= high)])
            pct = (count / num_patients) * 100
            f.write(f"  {label}: {count} patients ({pct:.2f}%)\n")
        
        # Detailed per-patient information
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED PER-PATIENT ANALYSIS\n")
        f.write("=" * 80 + "\n")
        display_cols = ['ID0', 'num_segments', 'ectopic_prevalence', 'PAC_count', 'PVC_count', 
                        'normal_count', 'ectopic_count', 'total_relevant']
        
        # Format the dataframe for text output
        f.write(f"{'ID0':<15} {'Segments':<12} {'Prevalence':<15} {'PAC':<8} {'PVC':<8} "
                f"{'Normal':<10} {'Ectopic':<10} {'Total':<10}\n")
        f.write("-" * 100 + "\n")
        
        for _, row in analysis_df.iterrows():
            f.write(f"{str(row['ID0']):<15} {int(row['num_segments']):<12} "
                    f"{row['ectopic_prevalence']:.4f} ({row['ectopic_prevalence']*100:.2f}%){'':<3} "
                    f"{int(row['PAC_count']):<8} {int(row['PVC_count']):<8} "
                    f"{int(row['normal_count']):<10} {int(row['ectopic_count']):<10} "
                    f"{int(row['total_relevant']):<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

def main():
    """Main analysis function."""
    print("=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    print(f"Dataset path: {DATASET_PATH}\n")
    
    # Load dataset
    df = load_dataset(DATASET_PATH)
    
    # Check available labels
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION (Overall)")
    print("=" * 80)
    label_counts = df['label'].value_counts()
    print(label_counts)
    print(f"\nTotal segments: {len(df)}")
    
    # Analyze patients and segments
    print("\n" + "=" * 80)
    print("PATIENT AND SEGMENT ANALYSIS")
    print("=" * 80)
    patient_stats = analyze_patients_and_segments(df)
    
    num_patients = len(patient_stats)
    total_segments = patient_stats['num_segments'].sum()
    
    print(f"Total number of patients: {num_patients}")
    print(f"Total number of segments: {total_segments}")
    print(f"\nSegments per patient statistics:")
    print(f"  Mean: {patient_stats['num_segments'].mean():.2f}")
    print(f"  Median: {patient_stats['num_segments'].median():.2f}")
    print(f"  Min: {patient_stats['num_segments'].min()}")
    print(f"  Max: {patient_stats['num_segments'].max()}")
    print(f"  Std: {patient_stats['num_segments'].std():.2f}")
    
    # Calculate ectopic prevalence per patient
    print("\n" + "=" * 80)
    print("ECTOPIC PREVALENCE PER PATIENT")
    print("=" * 80)
    print("Prevalence = (PAC + PVC) / (PAC + PVC + NORM)")
    
    prevalence_df = calculate_ectopic_prevalence_per_patient(df)
    
    # Reset index of patient_stats to make ID0 a column for merging
    patient_stats = patient_stats.reset_index()
    
    # Merge with patient stats
    analysis_df = patient_stats.merge(prevalence_df, on='ID0', how='left')
    
    # Print summary statistics
    print(f"\nEctopic prevalence statistics:")
    print(f"  Mean: {prevalence_df['ectopic_prevalence'].mean():.4f} ({prevalence_df['ectopic_prevalence'].mean()*100:.2f}%)")
    print(f"  Median: {prevalence_df['ectopic_prevalence'].median():.4f} ({prevalence_df['ectopic_prevalence'].median()*100:.2f}%)")
    print(f"  Min: {prevalence_df['ectopic_prevalence'].min():.4f} ({prevalence_df['ectopic_prevalence'].min()*100:.2f}%)")
    print(f"  Max: {prevalence_df['ectopic_prevalence'].max():.4f} ({prevalence_df['ectopic_prevalence'].max()*100:.2f}%)")
    print(f"  Std: {prevalence_df['ectopic_prevalence'].std():.4f} ({prevalence_df['ectopic_prevalence'].std()*100:.2f}%)")
    
    # Count patients by prevalence ranges
    print(f"\nPatients by ectopic prevalence range:")
    ranges = [
        (0.0, 0.0, "0% (No ectopics)"),
        (0.0, 0.1, "0-10%"),
        (0.1, 0.25, "10-25%"),
        (0.25, 0.5, "25-50%"),
        (0.5, 0.75, "50-75%"),
        (0.75, 1.0, "75-100%"),
        (1.0, 1.0, "100% (All ectopics)")
    ]
    
    for low, high, label in ranges:
        if low == high:
            count = len(prevalence_df[prevalence_df['ectopic_prevalence'] == low])
        else:
            count = len(prevalence_df[(prevalence_df['ectopic_prevalence'] > low) & 
                                      (prevalence_df['ectopic_prevalence'] <= high)])
        pct = (count / num_patients) * 100
        print(f"  {label}: {count} patients ({pct:.2f}%)")
    
    # Display detailed per-patient information (all patients)
    print("\n" + "=" * 80)
    print("DETAILED PER-PATIENT ANALYSIS (All patients)")
    print("=" * 80)
    display_cols = ['ID0', 'num_segments', 'ectopic_prevalence', 'PAC_count', 'PVC_count', 
                    'normal_count', 'ectopic_count', 'total_relevant']
    
    # Format for better table display
    print(f"{'ID0':<15} {'Segments':<12} {'Prevalence':<15} {'PAC':<8} {'PVC':<8} "
          f"{'Normal':<10} {'Ectopic':<10} {'Total':<10}")
    print("-" * 100)
    
    for _, row in analysis_df.iterrows():
        print(f"{str(row['ID0']):<15} {int(row['num_segments']):<12} "
              f"{row['ectopic_prevalence']:.4f} ({row['ectopic_prevalence']*100:.2f}%){'':<3} "
              f"{int(row['PAC_count']):<8} {int(row['PVC_count']):<8} "
              f"{int(row['normal_count']):<10} {int(row['ectopic_count']):<10} "
              f"{int(row['total_relevant']):<10}")
    
    # Save detailed results to CSV
    csv_output_path = DATASET_PATH.replace('.pkl', '_analysis.csv')
    analysis_df.to_csv(csv_output_path, index=False)
    print(f"\n✓ Saved detailed analysis (CSV) to: {csv_output_path}")
    
    # Save comprehensive text report
    txt_output_path = DATASET_PATH.replace('.pkl', '_analysis_report.txt')
    save_text_report(df, patient_stats, prevalence_df, analysis_df, txt_output_path)
    print(f"✓ Saved analysis report (TXT) to: {txt_output_path}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
