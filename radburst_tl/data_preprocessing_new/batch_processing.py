"""
Batch Processing Script for GAN Training Data Generation

This script reads the burst_list.csv and automatically processes all bursts
using the BurstFixedWindowSlicer, with separate output directories for different burst types.
"""

import pandas as pd
import os
from tqdm import tqdm
from slicing_utils_new import BurstFixedWindowSlicer


def load_burst_catalog(catalog_path):
    """
    Load and validate burst catalog
    
    Args:
        catalog_path (str): Path to burst_list.csv
        
    Returns:
        pandas.DataFrame: Validated burst catalog
    """
    print(f"📖 Loading burst catalog: {catalog_path}")
    
    # Load burst catalog
    burst_df = pd.read_csv(catalog_path)
    
    # Validation
    required_columns = ['file_name', 'start_time', 'end_time', 'type']
    missing_columns = [col for col in required_columns if col not in burst_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Basic statistics
    print(f"   Total bursts: {len(burst_df)}")
    print(f"   Date range: {burst_df['date'].min()} to {burst_df['date'].max()}")
    print(f"   Locations: {burst_df['location'].unique()}")
    
    # Type distribution
    type_counts = burst_df['type'].value_counts().sort_index()
    print(f"   Type distribution:")
    for burst_type, count in type_counts.items():
        print(f"     Type {burst_type}: {count} events ({count/len(burst_df)*100:.1f}%)")
    
    return burst_df


def setup_output_directories(base_dir):
    """
    Create separate directories for each burst type
    
    Args:
        base_dir (str): Base output directory
        
    Returns:
        dict: Dictionary mapping burst types to output directories
    """
    type_dirs = {}
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create type-specific directories
    burst_types = [2, 3, 5]  # Based on your data
    for burst_type in burst_types:
        type_dir = os.path.join(base_dir, f"type_{burst_type}")
        os.makedirs(type_dir, exist_ok=True)
        type_dirs[burst_type] = type_dir
        print(f"   📁 Created directory: {type_dir}")
    
    return type_dirs


def convert_catalog_to_burst_list(burst_df, original_csv_dir):
    """
    Convert burst catalog to format expected by process_multiple_bursts
    
    Args:
        burst_df: Burst catalog DataFrame
        original_csv_dir: Directory containing original CSV files
        
    Returns:
        dict: Dictionary with burst_list for each type
    """
    burst_lists_by_type = {}
    
    for _, row in burst_df.iterrows():
        burst_type = row['type']
        
        # Initialize list for this type if not exists
        if burst_type not in burst_lists_by_type:
            burst_lists_by_type[burst_type] = []
        
        # Create full path to CSV file
        csv_file_path = os.path.join(original_csv_dir, row['file_name'])
        
        # Create burst info dictionary
        burst_info = {
            'csv_file': csv_file_path,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'burst_type': burst_type,  # Add burst type at top level
            'metadata': {
                'date': row['date'],
                'location': row['location'], 
                'type': burst_type,
                'file_name': row['file_name']
            }
        }
        
        burst_lists_by_type[burst_type].append(burst_info)
    
    # Print summary
    print(f"\n📊 Burst Lists by Type:")
    for burst_type, burst_list in burst_lists_by_type.items():
        print(f"   Type {burst_type}: {len(burst_list)} bursts")
    
    return burst_lists_by_type


def process_all_bursts_by_type(catalog_path, original_csv_dir, output_base_dir, 
                             window_duration=4*60, overlap_ratio=0.5, apply_denoising=True,
                             cleaning_method="fast"):
    """
    Main function to process all bursts and separate by type
    
    Args:
        catalog_path (str): Path to burst_list.csv
        original_csv_dir (str): Directory containing original CSV files
        output_base_dir (str): Base directory for output
        window_duration (int): Window duration in seconds
        overlap_ratio (float): Overlap ratio for windows
        apply_denoising (bool): Whether to apply denoising
        cleaning_method (str): RFI cleaning method ('fast', 'comprehensive', 'conservative')
        
    Returns:
        dict: Processing results by type
    """
    print(f"\n{'='*70}")
    print(f"🚀 BATCH PROCESSING ALL BURSTS FOR GAN TRAINING")
    print(f"{'='*70}")
    
    # Load burst catalog
    burst_df = load_burst_catalog(catalog_path)
    
    # Setup output directories
    print(f"\n📁 Setting up output directories...")
    type_dirs = setup_output_directories(output_base_dir)
    
    # Convert to burst lists by type
    burst_lists_by_type = convert_catalog_to_burst_list(burst_df, original_csv_dir)
    
    # Initialize slicer
    slicer = BurstFixedWindowSlicer(
        window_duration=window_duration,
        overlap_ratio=overlap_ratio,
        target_size=(128, 128)
    )
    
    # Process each type separately
    results_by_type = {}
    
    for burst_type, burst_list in burst_lists_by_type.items():
        print(f"\n{'*'*50}")
        print(f"🎯 Processing Type {burst_type} Bursts ({len(burst_list)} events)")
        print(f"{'*'*50}")
        
        # Process all bursts of this type
        type_results = []
        total_windows = 0
        successful_bursts = 0
        
        for i, burst_info in enumerate(tqdm(burst_list, desc=f"Type {burst_type}")):
            try:
                result = slicer.slice_burst_with_fixed_windows(
                    csv_file_path=burst_info['csv_file'],
                    burst_start_time=burst_info['start_time'],
                    burst_end_time=burst_info['end_time'],
                    save_dir=type_dirs[burst_type],
                    apply_denoising=apply_denoising,
                    burst_type=burst_info['burst_type'],  # Pass burst type for advanced RFI cleaning
                    cleaning_method=cleaning_method  # Use specified cleaning method
                )
                
                # Add metadata to result
                result['burst_metadata'] = burst_info['metadata']
                type_results.append(result)
                total_windows += len(result['windows'])
                successful_bursts += 1
                
            except Exception as e:
                print(f"\n❌ Error processing burst {i+1} (Type {burst_type}): {e}")
                continue
        
        # Store results for this type
        results_by_type[burst_type] = {
            'results': type_results,
            'total_windows': total_windows,
            'successful_bursts': successful_bursts,
            'total_bursts': len(burst_list),
            'output_directory': type_dirs[burst_type]
        }
        
        print(f"\n✅ Type {burst_type} Processing Complete!")
        print(f"   Processed: {successful_bursts}/{len(burst_list)} bursts")
        print(f"   Generated: {total_windows} windows")
        print(f"   Saved to: {type_dirs[burst_type]}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"🎉 BATCH PROCESSING COMPLETED!")
    print(f"{'='*70}")
    
    total_processed_bursts = sum(r['successful_bursts'] for r in results_by_type.values())
    total_generated_windows = sum(r['total_windows'] for r in results_by_type.values())
    
    print(f"📊 Overall Summary:")
    print(f"   Total processed bursts: {total_processed_bursts}")
    print(f"   Total generated windows: {total_generated_windows}")
    print(f"   Output base directory: {output_base_dir}")
    
    print(f"\n📈 Results by Type:")
    for burst_type, result_info in results_by_type.items():
        success_rate = result_info['successful_bursts'] / result_info['total_bursts'] * 100
        avg_windows = result_info['total_windows'] / result_info['successful_bursts'] if result_info['successful_bursts'] > 0 else 0
        print(f"   Type {burst_type}: {result_info['successful_bursts']}/{result_info['total_bursts']} bursts "
              f"({success_rate:.1f}% success) → {result_info['total_windows']} windows "
              f"(avg: {avg_windows:.1f} windows/burst)")
    
    return results_by_type


def create_training_summary(results_by_type, output_file="training_data_summary.txt"):
    """
    Create a summary file of generated training data
    """
    with open(output_file, 'w') as f:
        f.write("🚀 GAN Training Data Generation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for burst_type, result_info in results_by_type.items():
            f.write(f"Type {burst_type} Solar Radio Bursts:\n")
            f.write(f"  Directory: {result_info['output_directory']}\n")
            f.write(f"  Processed bursts: {result_info['successful_bursts']}\n")
            f.write(f"  Generated windows: {result_info['total_windows']}\n")
            f.write(f"  Avg windows/burst: {result_info['total_windows']/result_info['successful_bursts'] if result_info['successful_bursts'] > 0 else 0:.1f}\n")
            f.write("\n")
            
            # List first few files for reference
            if result_info['results']:
                f.write(f"  Sample files:\n")
                for i, result in enumerate(result_info['results'][:3]):
                    f.write(f"    {i+1}. {result['metadata']['source_file']} → {len(result['windows'])} windows\n")
                f.write("\n")
    
    print(f"📝 Summary written to: {output_file}")


if __name__ == "__main__":
    # Configuration
    CATALOG_PATH = "/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/original/burst_list_240330_240729.csv"
    ORIGINAL_CSV_DIR = "/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/original"
    OUTPUT_BASE_DIR = "/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/gan_training_windows"
    
    # Process all bursts
    results = process_all_bursts_by_type(
        catalog_path=CATALOG_PATH,
        original_csv_dir=ORIGINAL_CSV_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        window_duration=4*60,  # 4 minutes
        overlap_ratio=0.5,     # 50% overlap
        apply_denoising=True
    )
    
    # Create summary
    create_training_summary(results, "gan_training_data_summary.txt")
    
    print(f"\n🎯 Ready for GAN Training!")
    print(f"   Type 2 data: {OUTPUT_BASE_DIR}/type_2/")
    print(f"   Type 3 data: {OUTPUT_BASE_DIR}/type_3/")  
    print(f"   Type 5 data: {OUTPUT_BASE_DIR}/type_5/")
