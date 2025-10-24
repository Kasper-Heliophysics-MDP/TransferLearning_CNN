#!/usr/bin/env python
"""
Compute Per-Frequency Moments for SpecGAN Training

This script computes mean and standard deviation for each frequency bin
across all CSV spectrogram files. Must be run BEFORE training SpecGAN.

Based on SpecGAN's moments computation workflow:
    python train_specgan.py moments ./train --data_dir ... --data_moments_fp moments.pkl

Usage:
    python compute_moments.py
    
Or with custom paths:
    python compute_moments.py --data_dir path/to/csvs --output moments.npz
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specgan.specgan_utils import compute_csv_moments


def main():
    """Main function to compute and save moments"""
    
    parser = argparse.ArgumentParser(
        description='Compute per-frequency moments for SpecGAN training'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/gan_training_windows_128/type_3/',
        help='Directory containing CSV spectrogram files (default: type_3 directory)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./checkpoints_specgan/type3_moments.npz',
        help='Output path for moments file (default: ./checkpoints_specgan/type3_moments.npz)'
    )
    
    parser.add_argument(
        '--all_types',
        action='store_true',
        help='If set, compute moments from all burst types (type_2, type_3, type_5)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("SpecGAN Per-Frequency Moments Computation")
    print("=" * 70)
    print(f"\n📁 Configuration:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Output file: {args.output}")
    print(f"   All types: {args.all_types}")
    print()
    
    # Use all types if requested
    if args.all_types:
        # Get parent directory (gan_training_windows_128)
        parent_dir = os.path.dirname(args.data_dir.rstrip('/'))
        if not os.path.basename(parent_dir) == 'gan_training_windows_128':
            parent_dir = '/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/gan_training_windows_128/'
        
        args.data_dir = parent_dir
        print(f"   Using all burst types from: {args.data_dir}")
    
    # Check if directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Directory not found: {args.data_dir}")
        print(f"\nPlease check the path or use --data_dir to specify correct path.")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"   Created output directory: {output_dir}")
    
    # Compute moments
    print("\n" + "=" * 70)
    print("Starting moments computation...")
    print("=" * 70)
    print()
    
    try:
        normalizer = compute_csv_moments(
            csv_dir=args.data_dir,
            output_path=args.output,
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\nMoments file saved to: {args.output}")
        print(f"\nYou can now train SpecGAN with:")
        print(f"   dataset = CSVSpectrogramDataset(")
        print(f"       root_dir='{args.data_dir}',")
        print(f"       normalize_method='per_frequency',")
        print(f"       moments_path='{args.output}',")
        print(f"       grayscale=True,")
        print(f"       augment=True")
        print(f"   )")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

