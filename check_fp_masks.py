#!/usr/bin/env python3
"""
Script to check if fp_mask files contain non-zero values.
"""
import numpy as np
import os

def check_mask_file(filepath, name):
    """Check if a mask file exists and contains non-zero values."""
    if not os.path.exists(filepath):
        print(f"❌ {name}: File not found at {filepath}")
        return False
    
    try:
        masks = np.load(filepath, allow_pickle=True)
        print(f"\n{'='*60}")
        print(f"Checking {name}")
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"Shape: {masks.shape}")
        print(f"Dtype: {masks.dtype}")
        
        # Check if all masks are zero
        total_elements = masks.size
        non_zero_elements = np.count_nonzero(masks)
        zero_elements = total_elements - non_zero_elements
        
        print(f"\nTotal elements: {total_elements:,}")
        print(f"Non-zero elements: {non_zero_elements:,} ({100*non_zero_elements/total_elements:.2f}%)")
        print(f"Zero elements: {zero_elements:,} ({100*zero_elements/total_elements:.2f}%)")
        
        # Check per sample
        n_samples = len(masks)
        samples_with_zeros = 0
        samples_with_nonzeros = 0
        
        for i in range(n_samples):
            mask = masks[i]
            if np.all(mask == 0):
                samples_with_zeros += 1
            else:
                samples_with_nonzeros += 1
        
        print(f"\nSamples with all zeros: {samples_with_zeros}/{n_samples} ({100*samples_with_zeros/n_samples:.2f}%)")
        print(f"Samples with non-zeros: {samples_with_nonzeros}/{n_samples} ({100*samples_with_nonzeros/n_samples:.2f}%)")
        
        # Check per channel
        if len(masks.shape) >= 3:
            print(f"\nPer-channel statistics:")
            for ch in range(4):
                channel_data = masks[:, :, ch] if masks.ndim == 3 else masks[:, ch]
                ch_nonzero = np.count_nonzero(channel_data)
                ch_total = channel_data.size
                print(f"  Channel {ch}: {ch_nonzero:,}/{ch_total:,} non-zero ({100*ch_nonzero/ch_total:.2f}%)")
        
        # Show a few example masks
        print(f"\nExample masks (first 3 samples):")
        for i in range(min(3, n_samples)):
            mask = masks[i]
            print(f"  Sample {i}: shape={mask.shape}, non-zero={np.count_nonzero(mask)}, max={np.max(mask)}, min={np.min(mask)}")
            if mask.ndim == 2:
                for ch in range(min(4, mask.shape[1])):
                    ch_data = mask[:, ch]
                    ch_nonzero = np.count_nonzero(ch_data)
                    print(f"    Channel {ch}: {ch_nonzero} non-zero values")
        
        if non_zero_elements == 0:
            print(f"\n❌ WARNING: {name} contains ALL ZEROS!")
            return False
        else:
            print(f"\n✓ {name} contains non-zero values")
            return True
            
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    base_dir = "PPG_data/splitted_data/sqi_50"
    
    files_to_check = [
        (os.path.join(base_dir, "fp_masks_train.npy"), "Train Masks"),
        (os.path.join(base_dir, "fp_masks_val.npy"), "Val Masks"),
        (os.path.join(base_dir, "fp_masks_test.npy"), "Test Masks"),
    ]
    
    all_ok = True
    for filepath, name in files_to_check:
        if not check_mask_file(filepath, name):
            all_ok = False
    
    print(f"\n{'='*60}")
    if all_ok:
        print("✓ All mask files contain non-zero values")
    else:
        print("❌ Some mask files are all zeros - investigation needed!")
    print(f"{'='*60}")

