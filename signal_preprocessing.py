"""
SIGNET Signal Preprocessing Module

This module provides functions for loading, cleaning, and preprocessing
plant electrical signal data for further analysis.

Author: Open Source Community
License: MIT
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def load_signal_data(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load plant electrical signal CSV files
    
    Args:
        file_paths: List of CSV file paths for plant electrical signal measurements
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Loaded dataframes for each measurement time
    """
    print("📁 Loading plant electrical signal data...")
    print("=" * 50)
    
    datasets = []
    measurement_times = ['morning_09h', 'noon_12h', 'evening_18h']
    
    for i, file_path in enumerate(file_paths):
        try:
            signal_data = pd.read_csv(file_path)
            datasets.append(signal_data)
            print(f"✅ {measurement_times[i]} data loaded: {signal_data.shape}")
        except Exception as e:
            print(f"❌ Failed to load file: {file_path}")
            print(f"Error: {e}")
            raise
    
    print("✅ Plant electrical signal data loading completed\n")
    return tuple(datasets)


def detect_signal_outliers(signal_data: pd.Series, iqr_multiplier: float = 2.0) -> Tuple[pd.Series, int]:
    """
    Detect and treat outliers in plant electrical signal using IQR method
    
    Args:
        signal_data: Plant electrical signal series to process
        iqr_multiplier: IQR multiplier for outlier detection threshold
        
    Returns:
        Tuple[pd.Series, int]: Processed signal data and number of outliers detected
    """
    # Calculate quartiles for signal analysis
    q1 = signal_data.quantile(0.25)
    q3 = signal_data.quantile(0.75)
    iqr = q3 - q1
    
    # Set outlier boundaries for electrical signal
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    # Create outlier mask for signal data
    outlier_mask = (signal_data < lower_bound) | (signal_data > upper_bound)
    
    # Apply linear interpolation only if outliers exist in signal
    if outlier_mask.sum() > 0:
        clean_signal = signal_data.copy()
        clean_signal[outlier_mask] = np.nan
        clean_signal = clean_signal.interpolate(method='linear')
        clean_signal = clean_signal.fillna(method='ffill').fillna(method='bfill')
    else:
        clean_signal = signal_data.copy()
    
    return clean_signal, outlier_mask.sum()


def preprocess_signals(morning_data: pd.DataFrame, noon_data: pd.DataFrame, evening_data: pd.DataFrame,
                      initial_trim: int = 1000, target_samples: int = 25000, 
                      iqr_multiplier: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess plant electrical signal time series data
    
    Args:
        morning_data, noon_data, evening_data: Plant electrical signal dataframes to preprocess
        initial_trim: Number of initial data points to remove for stabilization
        target_samples: Target number of samples for final analysis
        iqr_multiplier: IQR multiplier for outlier detection in electrical signals
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Preprocessed electrical signal dataframes
    """
    print("🔧 Starting plant electrical signal preprocessing...")
    print("=" * 80)
    
    # Plant electrical signal datasets
    signal_datasets = {
        'morning_09h': morning_data, 
        'noon_12h': noon_data, 
        'evening_18h': evening_data
    }
    processed_signals = {}
    
    for dataset_name, signal_df in signal_datasets.items():
        print(f"\n🔄 Processing {dataset_name} electrical signals (Shape: {signal_df.shape})")
        print("-" * 50)
        
        # Step 1: Trim initial unstable electrical signal measurements
        start_idx = initial_trim
        end_idx = start_idx + target_samples
        
        # Validate electrical signal data length
        if len(signal_df) < (initial_trim + target_samples):
            print(f"⚠️  Warning: Insufficient signal data. Required: {initial_trim + target_samples}, Available: {len(signal_df)}")
            end_idx = len(signal_df)
        
        # Slice electrical signal data and reset index
        trimmed_signals = signal_df.iloc[start_idx:end_idx].copy()
        trimmed_signals.reset_index(drop=True, inplace=True)
        print(f"📏 Signal length adjusted: {len(signal_df)} → {len(trimmed_signals)}")
        
        # Step 2: Process outliers for each electrical signal channel
        clean_signals = trimmed_signals.copy()
        total_outliers = 0
        
        print(f"🔍 Outlier detection results ({iqr_multiplier}×IQR):")
        
        for channel in trimmed_signals.columns:
            processed_channel, outlier_count = detect_signal_outliers(trimmed_signals[channel], iqr_multiplier)
            clean_signals[channel] = processed_channel
            total_outliers += outlier_count
            
            # Output results for electrical signal channel
            outlier_percentage = outlier_count / len(trimmed_signals) * 100
            print(f"  📊 {channel}: {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        # Overall summary for electrical signal processing
        total_points = len(trimmed_signals) * len(trimmed_signals.columns)
        print(f"📈 Total outliers: {total_outliers}/{total_points} ({total_outliers/total_points*100:.3f}%)")
        
        # Store processed electrical signals
        processed_signals[dataset_name] = clean_signals
    
    return processed_signals['morning_09h'], processed_signals['noon_12h'], processed_signals['evening_18h']


def combine_signal_measurements(morning_signals: pd.DataFrame, noon_signals: pd.DataFrame, 
                               evening_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Combine plant electrical signal measurements from different time periods
    
    Args:
        morning_signals, noon_signals, evening_signals: Processed electrical signal dataframes
        
    Returns:
        pd.DataFrame: Combined plant electrical signal dataset
    """
    print(f"\n{'='*80}")
    print("🔗 Combining plant electrical signal measurements")
    print(f"{'='*80}")
    
    # Combine all electrical signal measurements
    combined_signals = pd.concat([morning_signals, noon_signals, evening_signals], ignore_index=True)
    
    print(f"✅ Signal combination completed: {combined_signals.shape}")
    
    return combined_signals


def apply_signal_scaling(signal_df: pd.DataFrame, sigma: float = 0.1) -> pd.DataFrame:
    """
    Apply data augmentation scaling to signal DataFrame
    
    Args:
        signal_df: Input signal DataFrame
        sigma: Scaling noise level
        
    Returns:
        pd.DataFrame: Scaled signal DataFrame
    """
    print(f"🔄 Applying signal scaling with sigma={sigma}")
    
    # Generate scaling factors for each column (signal channel)
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, len(signal_df.columns)))
    
    # Create scaling matrix matching DataFrame dimensions
    scaling_matrix = np.tile(scaling_factor, (len(signal_df), 1))
    
    # Apply scaling to DataFrame values
    scaled_values = signal_df.values * scaling_matrix
    
    # Create new DataFrame with scaled values and original structure
    scaled_df = pd.DataFrame(scaled_values, columns=signal_df.columns, index=signal_df.index)
    
    print(f"✅ Signal scaling completed")
    
    return scaled_df


if __name__ == "__main__":
    print("🔧 SIGNET Signal Preprocessing Module")
    print("Available functions:")
    print("- load_signal_data()")
    print("- preprocess_signals()")
    print("- combine_signal_measurements()")
    print("- apply_signal_scaling()")