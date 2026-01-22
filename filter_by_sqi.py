"""
Filter splitted_data by SQI threshold and save to separate folders.
"""
import os
import numpy as np
import shutil

def filter_by_sqi(input_dir, output_base_dir, sqi_thresholds):
    """
    Filter all .npy files by SQI threshold and save to separate folders.
    
    Args:
        input_dir: Path to sqi_0 folder
        output_base_dir: Base path for output (e.g., splitted_data)
        sqi_thresholds: List of SQI thresholds to filter by
    """
    splits = ['train', 'val', 'test']
    
    for threshold in sqi_thresholds:
        output_dir = os.path.join(output_base_dir, f'sqi_{threshold}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=== Processing SQI >= {threshold} ===")
        
        for split in splits:
            # Load SQI values for this split
            sqi_file = os.path.join(input_dir, f'sqi_{split}.npy')
            sqi_values = np.load(sqi_file, allow_pickle=True)
            
            # Create mask for samples with SQI >= threshold
            mask = sqi_values >= threshold
            n_kept = np.sum(mask)
            print(f"  {split}: {n_kept}/{len(sqi_values)} samples (SQI >= {threshold})")
            
            # Find all .npy files for this split and filter them
            for filename in os.listdir(input_dir):
                # Match files like xxx_train.npy or ppg_train_normalized.npy
                if f'_{split}.npy' in filename or f'_{split}_' in filename:
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    
                    data = np.load(input_path, allow_pickle=True)
                    filtered_data = data[mask]
                    np.save(output_path, filtered_data)
        
        # Copy non-.npy files (csv, json, txt) as-is or regenerate if needed
        for filename in os.listdir(input_dir):
            if not filename.endswith('.npy'):
                src = os.path.join(input_dir, filename)
                dst = os.path.join(output_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        print(f"  Saved to: {output_dir}")


if __name__ == '__main__':
    input_dir = 'PPG_data/splitted_data/sqi_0'
    output_base_dir = 'PPG_data/splitted_data'
    sqi_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    filter_by_sqi(input_dir, output_base_dir, sqi_thresholds)
    print("\nDone!")

