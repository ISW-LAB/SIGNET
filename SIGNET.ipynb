# =============================================================================
# SIGNET: Plant Electrical Signal Classification Pipeline
# 
# This notebook provides an end-to-end pipeline for:
# 1. Plant electrical signal preprocessing
# 2. Time series to image conversion (MTF, GAF, RP)
# 3. Multi-modal deep learning classification
# 
# Author: Open Source Community
# License: MIT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import platform
import os
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from PIL import Image
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Signal Processing Libraries
from scipy.stats import norm

print("🚀 SIGNET Pipeline initialized successfully!")
print("📊 Available modules: Signal Preprocessing, Image Encoding, Deep Learning")

# =============================================================================
# SECTION 1: SIGNAL PREPROCESSING MODULE
# =============================================================================

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
    # Generate scaling factors for each column (signal channel)
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, len(signal_df.columns)))
    
    # Create scaling matrix matching DataFrame dimensions
    scaling_matrix = np.tile(scaling_factor, (len(signal_df), 1))
    
    # Apply scaling to DataFrame values
    scaled_values = signal_df.values * scaling_matrix
    
    # Create new DataFrame with scaled values and original structure
    scaled_df = pd.DataFrame(scaled_values, columns=signal_df.columns, index=signal_df.index)
    
    return scaled_df


# =============================================================================
# SECTION 2: SIGNAL-TO-IMAGE ENCODING MODULE
# =============================================================================

class SaveFormat(Enum):
    """Enumeration for output file formats."""
    PNG = "png"
    NUMPY = "npy"
    BOTH = "both"


@dataclass
class SignalStats:
    """Container for signal statistics used in normalization."""
    mean: float
    std: float
    min: float
    max: float


class SAXBreakpoints:
    """SAX (Symbolic Aggregate approXimation) breakpoint calculator for normal distribution."""
    
    # Predefined breakpoints for common alphabet sizes
    _PREDEFINED_BREAKPOINTS = {
        2: [0],
        3: [-0.43, 0.43],
        4: [-0.67, 0, 0.67],
        5: [-0.84, -0.25, 0.25, 0.84],
        6: [-0.97, -0.43, 0, 0.43, 0.97],
        7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
        8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
    }
    
    @classmethod
    def get_breakpoints(cls, alphabet_size: int) -> List[float]:
        """Get SAX breakpoints for given alphabet size."""
        if alphabet_size in cls._PREDEFINED_BREAKPOINTS:
            return cls._PREDEFINED_BREAKPOINTS[alphabet_size]
        
        # General case using percentiles for arbitrary alphabet sizes
        percentiles = np.linspace(0, 100, alphabet_size + 1)[1:-1]
        return [norm.ppf(p/100) for p in percentiles]


class BaseSignalEncoder:
    """Base class for time series to image encodings."""
    
    def __init__(self, window_size: int = 64):
        """Initialize base encoder."""
        self.window_size = window_size
        self.signal_stats: Optional[SignalStats] = None
        
    def fit_signal_data(self, signal_data_list: List[Union[pd.DataFrame, np.ndarray]]) -> None:
        """Compute global statistics from signal data for normalization."""
        all_values = []
        
        for signal_data in signal_data_list:
            if isinstance(signal_data, pd.DataFrame):
                for column in signal_data.columns:
                    values = signal_data[column].values.astype(float).flatten()
                    all_values.extend(values[np.isfinite(values)])
            else:
                values = np.array(signal_data, dtype=float).flatten()
                all_values.extend(values[np.isfinite(values)])
        
        all_values = np.array(all_values, dtype=float)
        all_values = all_values[np.isfinite(all_values)]
        
        if len(all_values) == 0:
            raise ValueError("No valid values found in signal data")
        
        self.signal_stats = SignalStats(
            mean=np.mean(all_values),
            std=np.std(all_values),
            min=np.min(all_values),
            max=np.max(all_values)
        )
        
        print(f"📈 {self.__class__.__name__}: mean={self.signal_stats.mean:.2f}, "
              f"std={self.signal_stats.std:.2f}")

    def encode_window(self, window_data: np.ndarray) -> np.ndarray:
        """Encode window data to image matrix."""
        raise NotImplementedError("Subclasses must implement encode_window method")


class MTF_Encoder(BaseSignalEncoder):
    """Markov Transition Field implementation with SAX encoding."""
    
    def __init__(self, window_size: int = 64, quantization_levels: int = 8):
        """Initialize MTF encoder."""
        super().__init__(window_size)
        self.quantization_levels = quantization_levels
        self.breakpoints = SAXBreakpoints.get_breakpoints(quantization_levels)
        self.transition_matrix: Optional[np.ndarray] = None
    
    def fit_signal_data(self, signal_data_list: List[Union[pd.DataFrame, np.ndarray]]) -> None:
        """Fit MTF with global statistics and transition matrix."""
        super().fit_signal_data(signal_data_list)
        self.transition_matrix = self._compute_transition_matrix(signal_data_list)
    
    def _quantize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply SAX-based quantization to time series."""
        if self.signal_stats is None:
            # Use local statistics if global not available
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        else:
            # Use global statistics for consistent quantization
            signal_norm = (signal - self.signal_stats.mean) / (self.signal_stats.std + 1e-8)
        
        quantized = np.zeros(len(signal_norm), dtype=int)
        for i, val in enumerate(signal_norm):
            symbol_idx = sum(1 for bp in self.breakpoints if val >= bp)
            quantized[i] = min(symbol_idx, self.quantization_levels - 1)
        
        return quantized
    
    def _compute_transition_matrix(self, signal_data_list: List[Union[pd.DataFrame, np.ndarray]]) -> np.ndarray:
        """Compute global transition matrix from signal data."""
        transition_counts = np.zeros((self.quantization_levels, self.quantization_levels))
        
        for signal_data in signal_data_list:
            if isinstance(signal_data, pd.DataFrame):
                for column in signal_data.columns:
                    signal = signal_data[column].values.astype(float)
                    quantized = self._quantize_signal(signal)
                    # Count state transitions
                    for i in range(len(quantized) - 1):
                        transition_counts[quantized[i], quantized[i + 1]] += 1
            else:
                signal = np.array(signal_data, dtype=float).flatten()
                quantized = self._quantize_signal(signal)
                for i in range(len(quantized) - 1):
                    transition_counts[quantized[i], quantized[i + 1]] += 1
        
        # Normalize to get transition probabilities
        row_sums = transition_counts.sum(axis=1)
        for i in range(self.quantization_levels):
            if row_sums[i] > 0:
                transition_counts[i, :] /= row_sums[i]
        
        return transition_counts
    
    def encode_window(self, window_data: np.ndarray) -> np.ndarray:
        """Compute MTF matrix for the given window."""
        if self.transition_matrix is None:
            raise ValueError("MTF not fitted. Call fit_signal_data first.")
        
        n = len(window_data)
        quantized = self._quantize_signal(window_data)
        
        # Build MTF matrix using transition probabilities
        mtf_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mtf_matrix[i, j] = self.transition_matrix[quantized[i], quantized[j]]
        
        return mtf_matrix


class GAF_Encoder(BaseSignalEncoder):
    """Gramian Angular Field implementation."""
    
    def __init__(self, window_size: int = 64, scale_factor: float = 1.2):
        """Initialize GAF encoder."""
        super().__init__(window_size)
        self.scale_factor = scale_factor
        self.upper_bound: Optional[float] = None
        self.lower_bound: Optional[float] = None
    
    def fit_signal_data(self, signal_data_list: List[Union[pd.DataFrame, np.ndarray]]) -> None:
        """Fit GAF with global bounds for consistent normalization."""
        super().fit_signal_data(signal_data_list)
        
        self.upper_bound = self.signal_stats.max * self.scale_factor
        self.lower_bound = self.signal_stats.min * self.scale_factor
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize time series to [-1, 1] using global bounds."""
        if self.upper_bound is None or self.lower_bound is None:
            raise ValueError("GAF not fitted. Call fit_signal_data first.")
        
        if self.upper_bound == self.lower_bound:
            return np.zeros_like(signal)
        
        signal_clipped = np.clip(signal, self.lower_bound, self.upper_bound)
        signal_norm = 2 * (signal_clipped - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1
        return signal_norm
    
    def encode_window(self, window_data: np.ndarray) -> np.ndarray:
        """Compute GAF matrix for the given window."""
        signal_norm = self._normalize_signal(window_data)
        # Convert to angular representation
        phi = np.arccos(np.clip(signal_norm, -1, 1))
        # Compute Gramian angular field
        cos_sum = np.cos(phi[:, np.newaxis] + phi[np.newaxis, :])
        return cos_sum


class RP_Encoder(BaseSignalEncoder):
    """Recurrence Plot implementation."""
    
    def __init__(self, window_size: int = 64, add_mean: bool = True):
        """Initialize RP encoder."""
        super().__init__(window_size)
        self.add_mean = add_mean
    
    def encode_window(self, window_data: np.ndarray) -> np.ndarray:
        """Compute RP matrix for the given window."""
        # Compute pairwise absolute differences
        diff_matrix = window_data[:, np.newaxis] - window_data[np.newaxis, :]
        rp_matrix = np.abs(diff_matrix)
        
        if self.add_mean:
            # Add mean to shift values for better visualization
            rp_matrix = rp_matrix + np.mean(window_data)
        
        return rp_matrix


def create_signal_windows(signal_data: np.ndarray, window_size: int, stride: int = 1) -> List[np.ndarray]:
    """Create sliding windows from time series data."""
    windows = []
    for i in range(0, len(signal_data) - window_size + 1, stride):
        windows.append(signal_data[i:i + window_size])
    return windows


def save_encoded_images(matrix: np.ndarray, 
                       filepath: str, 
                       save_format: SaveFormat = SaveFormat.PNG, 
                       colormap: str = 'viridis', 
                       dpi: int = 100, 
                       image_base_dir: Optional[str] = None, 
                       numpy_base_dir: Optional[str] = None) -> None:
    """Save encoded matrix as PNG image and/or NumPy file."""
    
    if save_format in [SaveFormat.PNG, SaveFormat.BOTH]:
        # Determine PNG filepath
        if image_base_dir is not None:
            png_filepath = os.path.join(image_base_dir, filepath + ".png")
            os.makedirs(os.path.dirname(png_filepath), exist_ok=True)
        else:
            png_filepath = f"{filepath}.png"
        
        # Save as PNG image without axes and padding
        plt.figure(figsize=(matrix.shape[0]/dpi, matrix.shape[1]/dpi), dpi=dpi)
        plt.imshow(matrix, cmap=colormap, aspect='auto')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(png_filepath, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    
    if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH]:
        # Determine NumPy filepath
        if numpy_base_dir is not None:
            npy_filepath = os.path.join(numpy_base_dir, filepath + ".npy")
            os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)
        else:
            npy_filepath = f"{filepath}.npy"
        
        # Save as NumPy file
        np.save(npy_filepath, matrix)


def convert_signals_to_images(signal_df: pd.DataFrame, 
                             signal_columns: List[str],
                             window_size: int = 64, 
                             stride: int = 32, 
                             output_dir: str = 'encoded_images',
                             save_format: SaveFormat = SaveFormat.PNG,
                             colormap: str = 'viridis',
                             dpi: int = 100,
                             image_base_dir: Optional[str] = None,
                             numpy_base_dir: Optional[str] = None) -> Dict:
    """
    Convert time series signals to encoded images using MTF, GAF, and RP methods.
    """
    
    # Determine actual output directories
    if image_base_dir is not None:
        actual_image_dir = os.path.join(image_base_dir, output_dir)
    else:
        actual_image_dir = output_dir
    
    if numpy_base_dir is not None:
        actual_numpy_dir = os.path.join(numpy_base_dir, output_dir)
    else:
        actual_numpy_dir = output_dir
    
    # Create output directories
    if save_format in [SaveFormat.PNG, SaveFormat.BOTH]:
        os.makedirs(actual_image_dir, exist_ok=True)
    if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH]:
        os.makedirs(actual_numpy_dir, exist_ok=True)
    
    # Initialize encoders
    encoders = {
        'MTF': MTF_Encoder(window_size=window_size),
        'GAF': GAF_Encoder(window_size=window_size),
        'RP': RP_Encoder(window_size=window_size),   
    }
    
    print(f"🚀 Fitting encoders to signal data...")
    print(f"📁 Save format: {save_format.value}")
    if save_format in [SaveFormat.PNG, SaveFormat.BOTH]:
        print(f"🖼️  PNG images will be saved to: {actual_image_dir}")
    if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH]:
        print(f"🔢 NumPy files will be saved to: {actual_numpy_dir}")
    
    # Prepare signal data for fitting encoders
    signal_data_list = [signal_df[signal_columns]]
    
    # Fit encoders with signal data
    for name, encoder in encoders.items():
        try:
            encoder.fit_signal_data(signal_data_list)
        except Exception as e:
            print(f"⚠️  Warning: Failed to fit {name}: {e}")
    
    print(f"\n📊 Processing {len(signal_columns)} signal channels...")
    
    # Statistics tracking
    total_files = 0
    channel_stats = {}
    
    # Process each signal channel
    for channel_idx, column in enumerate(signal_columns):
        print(f"\n🔄 Processing channel {channel_idx + 1}: {column}")
        
        # Get channel data
        channel_data = signal_df[column].values.astype(float)
        
        # Create sliding windows
        windows = create_signal_windows(channel_data, window_size, stride)
        print(f"  📈 Created {len(windows)} windows")
        
        # Create channel directory structure
        channel_name = f'CH-{103+channel_idx}'
        
        # Create channel directories for both image and numpy
        if save_format in [SaveFormat.PNG, SaveFormat.BOTH]:
            channel_image_dir = os.path.join(actual_image_dir, channel_name)
            os.makedirs(channel_image_dir, exist_ok=True)
        
        if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH]:
            channel_numpy_dir = os.path.join(actual_numpy_dir, channel_name)
            os.makedirs(channel_numpy_dir, exist_ok=True)
        
        # Create encoder subdirectories and initialize statistics
        encoder_stats = {}
        for encoder_name in encoders.keys():
            if save_format in [SaveFormat.PNG, SaveFormat.BOTH]:
                encoder_image_dir = os.path.join(channel_image_dir, encoder_name)
                os.makedirs(encoder_image_dir, exist_ok=True)
            
            if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH]:
                encoder_numpy_dir = os.path.join(channel_numpy_dir, encoder_name)
                os.makedirs(encoder_numpy_dir, exist_ok=True)
            
            encoder_stats[encoder_name] = 0
        
        # Process each window with each encoder
        for window_idx, window_data in enumerate(windows):
            if window_idx % 100 == 0:
                print(f"  🔢 Processing window {window_idx + 1}/{len(windows)}")
            
            for encoder_name, encoder in encoders.items():
                try:
                    # Encode window data
                    encoded_matrix = encoder.encode_window(window_data)
                    
                    # Prepare base filepath (without extension)
                    filename_base = f'window_{window_idx:04d}'
                    
                    # Use the original output_dir structure for relative path
                    base_filepath = os.path.join(output_dir, channel_name, encoder_name, filename_base)
                    
                    # Save encoded data with separate base directories
                    save_encoded_images(
                        matrix=encoded_matrix, 
                        filepath=base_filepath, 
                        save_format=save_format, 
                        colormap=colormap, 
                        dpi=dpi,
                        image_base_dir=image_base_dir,
                        numpy_base_dir=numpy_base_dir
                    )
                    
                    encoder_stats[encoder_name] += 1
                    total_files += 1 if save_format != SaveFormat.BOTH else 2
                    
                except Exception as e:
                    print(f"    ⚠️  Error encoding window {window_idx} with {encoder_name}: {e}")
                    continue
        
        channel_stats[channel_name] = encoder_stats
    
    # Print completion summary
    print(f"\n✅ Signal-to-image conversion completed!")
    if save_format in [SaveFormat.PNG, SaveFormat.BOTH]:
        print(f"🖼️  PNG images saved in: {actual_image_dir}")
    if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH]:
        print(f"🔢 NumPy files saved in: {actual_numpy_dir}")
    
    # Print detailed statistics
    for channel_name, encoder_stats in channel_stats.items():
        for encoder_name, count in encoder_stats.items():
            extension = ""
            if save_format == SaveFormat.PNG:
                extension = "PNG images"
            elif save_format == SaveFormat.NUMPY:
                extension = "NumPy files"
            elif save_format == SaveFormat.BOTH:
                extension = "files (PNG + NumPy)"
            
            print(f"  📁 {channel_name}/{encoder_name}: {count} {extension}")
    
    print(f"  🎯 Total files generated: {total_files}")
    
    return {
        'image_dir': actual_image_dir if save_format in [SaveFormat.PNG, SaveFormat.BOTH] else None,
        'numpy_dir': actual_numpy_dir if save_format in [SaveFormat.NUMPY, SaveFormat.BOTH] else None,
        'total_files': total_files,
        'channel_stats': channel_stats
    }


# =============================================================================
# SECTION 3: DEEP LEARNING MODEL MODULE
# =============================================================================

class SIGNET_Dataset(Dataset):
    """Dataset for loading MTF, GAF, RP images"""
    def __init__(self, root_dir, transform=None, target_encodings=['MTF', 'GAF', 'RP', 'MTF_scaling', 'GAF_scaling', 'RP_scaling']):
        self.root_dir = root_dir
        self.transform = transform
        self.target_encodings = target_encodings
        self.data = []
        self.class_to_idx = {'irrigated_1': 0, 'irrigated_2': 1, 'irrigated_3': 2}
        self.idx_to_class = {0: 'irrigated_1', 1: 'irrigated_2', 2: 'irrigated_3'}
        
        self._load_dataset()
    
    def _load_dataset(self):
        for class_name in ['irrigated_1', 'irrigated_2', 'irrigated_3']:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️  Warning: {class_path} does not exist")
                continue
            
            # Collect samples based on the first encoding
            first_encoding = self.target_encodings[0]
            first_encoding_path = os.path.join(class_path, first_encoding)
            
            if not os.path.exists(first_encoding_path):
                print(f"⚠️  Warning: {first_encoding_path} does not exist")
                continue
            
            # Base image files
            base_images = [f for f in os.listdir(first_encoding_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in base_images:
                sample_paths = {}
                # Collect images from each encoding with the same filename
                for encoding in self.target_encodings:
                    encoding_path = os.path.join(class_path, encoding)
                    img_path = os.path.join(encoding_path, img_file)
                    
                    if os.path.exists(img_path):
                        sample_paths[encoding] = img_path
                
                # Add only if all encoding images exist
                if len(sample_paths) == len(self.target_encodings):
                    label = self.class_to_idx[class_name]
                    self.data.append((sample_paths, label, class_name))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_paths, label, class_name = self.data[idx]
        
        # Load images for each encoding (order guaranteed)
        images = []
        for encoding in self.target_encodings:
            try:
                image = Image.open(sample_paths[encoding]).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"❌ Error loading image {sample_paths[encoding]}: {e}")
                
                if self.transform:
                    error_image = self.transform(Image.new('RGB', (224, 224), color='black'))
                else:
                    error_image = torch.zeros(3, 224, 224)
                images.append(error_image)
        
        return images, label


def get_image_transforms():
    """Data preprocessing transforms for images"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block for MobileNet V2 architecture"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # Expansion layer (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise conv (3x3)
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution (1x1)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class CNN_Backbone(nn.Module):
    """MobileNet V2 Backbone for feature extraction"""
    def __init__(self, width_mult=1.0, pretrained=True):
        super(CNN_Backbone, self).__init__()
        
        # First convolution layer
        input_channel = int(32 * width_mult)
        self.features = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted Residual blocks configuration
        # [expand_ratio, channels, num_blocks, stride]
        inverted_residual_setting = [
            [1, 16, 1, 1],   # stage 1
            [6, 24, 2, 2],   # stage 2
            [6, 32, 3, 2],   # stage 3
            [6, 64, 4, 2],   # stage 4
            [6, 96, 3, 1],   # stage 5
            [6, 160, 3, 2],  # stage 6
            [6, 320, 1, 1],  # stage 7
        ]
        
        # Build Inverted Residual blocks
        features = [self.features]
        
        for expand_ratio, channels, num_blocks, stride in inverted_residual_setting:
            output_channel = int(channels * width_mult)
            
            for i in range(num_blocks):
                if i == 0:
                    features.append(InvertedResidualBlock(input_channel, output_channel, stride, expand_ratio))
                else:
                    features.append(InvertedResidualBlock(input_channel, output_channel, 1, expand_ratio))
                input_channel = output_channel
        
        # Final convolution layer
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        features.extend([
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ])
        
        self.features = nn.Sequential(*features)
        
        # Load ImageNet pretrained weights
        if pretrained:
            self._load_pretrained_weights()
        else:
            self._initialize_weights()
    
    def _load_pretrained_weights(self):
        """Load ImageNet pretrained weights"""
        try:
            # Load torchvision's pretrained MobileNet V2
            pretrained_model = models.mobilenet_v2(pretrained=True)
            
            # Copy only features part weights
            pretrained_features = pretrained_model.features
            
            # Copy state_dict
            own_state = self.features.state_dict()
            pretrained_state = pretrained_features.state_dict()
            
            # Load only matching weights
            for name, param in pretrained_state.items():
                if name in own_state:
                    if own_state[name].shape == param.shape:
                        own_state[name].copy_(param)
            
            print("✅ Pretrained weights loaded successfully!")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained weights: {e}")
            print("🔄 Initializing with random weights...")
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class SIGNET_Model(nn.Module):
    """Multi-modal classification model using late fusion"""
    def __init__(self, num_classes=3, num_modalities=6, dropout_rate=0.5, pretrained=True):
        super(SIGNET_Model, self).__init__()
        self.num_modalities = num_modalities
        self.dropout_rate = dropout_rate
        
        # Create backbone for each modality
        self.backbones = nn.ModuleList()
        for _ in range(num_modalities):
            backbone = CNN_Backbone(pretrained=pretrained)
            # Add GAP and Flatten
            backbone = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.backbones.append(backbone)
        
        # Feature dimension
        self.feature_dim = 1280
        
        # 3-layer MLP fusion classifier
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim * num_modalities, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features from each modality independently
        features = []
        for i, backbone in enumerate(self.backbones):
            feature = backbone(x[i])
            features.append(feature)
        
        # Concatenate features and classify
        concatenated_features = torch.cat(features, dim=1)
        return self.fusion_classifier(concatenated_features)


def train_model_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc='🏋️  Training'):
        # Move data to device
        images = [img.to(device) for img in images]
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='🧪 Evaluating'):
            # Move data to device
            images = [img.to(device) for img in images]
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_model_checkpoint(model, optimizer, epoch, loss, acc, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }, path)
    print(f"💾 Model checkpoint saved: {path}")


def load_model_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['accuracy']
    print(f"📂 Model checkpoint loaded: {path}")
    return model, optimizer, epoch, loss, acc


def train_signet_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001, save_dir='./models'):
    """Main training function with 80/20 train/test split"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Dataset preparation
    print("📊 Loading dataset...")
    transform = get_image_transforms()
    
    # Load full dataset
    full_dataset = SIGNET_Dataset(data_dir, transform=transform)
    
    # 80/20 train/test split
    total_size = len(full_dataset)
    test_size = int(total_size * 0.2)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📈 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = SIGNET_Model(num_classes=3, num_modalities=6, dropout_rate=0.5, pretrained=True)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    best_train_acc = 0.0
    os.makedirs(save_dir, exist_ok=True)
    
    # Start training
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_model_epoch(model, train_loader, criterion, optimizer, device)
        print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Save best model based on training accuracy
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            save_model_checkpoint(model, optimizer, epoch, train_loss, train_acc, 
                      os.path.join(save_dir, 'best_model.pth'))
            print(f"🏆 New best model saved! Train Acc: {train_acc:.4f}")
    
    print(f"\n✅ Training completed! Best training accuracy: {best_train_acc:.4f}")
    
    # Test the final model
    print("\n🧪 Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
    
    # Print test results
    print(f"\n📋 Test Results:")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    class_names = ['irrigated_1', 'irrigated_2', 'irrigated_3']
    print("\n📊 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    return model, test_loss, test_acc


def test_trained_model(model_path, test_data_dir, batch_size=32):
    """Test model on separate test dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model = SIGNET_Model(num_classes=3, num_modalities=6, dropout_rate=0.5, pretrained=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Test dataset
    transform = get_image_transforms()
    test_dataset = SIGNET_Dataset(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # Evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
    
    # Print results
    print(f"\n📋 Test Results:")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"🎯 Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    class_names = ['irrigated_1', 'irrigated_2', 'irrigated_3']
    print("\n📊 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    return test_loss, test_acc


# =============================================================================
# SECTION 4: INTEGRATED PIPELINE EXECUTION
# =============================================================================

def run_complete_pipeline(
    # Signal preprocessing parameters
    signal_file_paths: List[str],
    initial_trim: int = 1000,
    target_samples: int = 25000,
    iqr_multiplier: float = 2.0,
    scaling_sigma: float = 0.1,
    
    # Signal-to-image parameters
    window_size: int = 64,
    stride: int = 32,
    save_format: SaveFormat = SaveFormat.PNG,
    colormap: str = 'viridis',
    
    # Model training parameters
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    
    # Directory parameters
    output_base_dir: str = './signet_output',
    signal_output_dir: str = 'processed_signals',
    image_output_dir: str = 'encoded_images',
    model_output_dir: str = 'trained_models'
):
    """
    Run the complete SIGNET pipeline from raw signals to trained model
    
    Args:
        signal_file_paths: List of paths to signal CSV files
        All other parameters: Configuration for each pipeline step
        
    Returns:
        Dictionary containing results from each pipeline step
    """
    
    print("🌟" + "="*80)
    print("🌟 SIGNET: Complete Plant Electrical Signal Classification Pipeline")
    print("🌟" + "="*80)
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    processed_signal_dir = os.path.join(output_base_dir, signal_output_dir)
    image_dir = os.path.join(output_base_dir, image_output_dir)
    model_dir = os.path.join(output_base_dir, model_output_dir)
    
    os.makedirs(processed_signal_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    results = {}
    
    # =============================================================================
    # STEP 1: SIGNAL PREPROCESSING
    # =============================================================================
    print(f"\n{'🔧 STEP 1: SIGNAL PREPROCESSING'}")
    print("="*80)
    
    # Load signal data
    morning_signals, noon_signals, evening_signals = load_signal_data(signal_file_paths)
    
    # Preprocess signals
    morning_processed, noon_processed, evening_processed = preprocess_signals(
        morning_signals, noon_signals, evening_signals,
        initial_trim=initial_trim,
        target_samples=target_samples,
        iqr_multiplier=iqr_multiplier
    )
    
    # Combine signals
    combined_signals = combine_signal_measurements(morning_processed, noon_processed, evening_processed)
    
    # Apply scaling for augmentation
    scaled_signals = apply_signal_scaling(combined_signals, sigma=scaling_sigma)
    
    # Save processed signals
    original_signal_path = os.path.join(processed_signal_dir, 'combined_signals.csv')
    scaled_signal_path = os.path.join(processed_signal_dir, 'scaled_signals.csv')
    
    combined_signals.to_csv(original_signal_path, index=False)
    scaled_signals.to_csv(scaled_signal_path, index=False)
    
    print(f"💾 Original signals saved: {original_signal_path}")
    print(f"💾 Scaled signals saved: {scaled_signal_path}")
    
    results['preprocessing'] = {
        'combined_signals': combined_signals,
        'scaled_signals': scaled_signals,
        'original_signal_path': original_signal_path,
        'scaled_signal_path': scaled_signal_path
    }
    
    # =============================================================================
    # STEP 2: SIGNAL-TO-IMAGE CONVERSION
    # =============================================================================
    print(f"\n{'🖼️ STEP 2: SIGNAL-TO-IMAGE CONVERSION'}")
    print("="*80)
    
    # Get signal columns
    signal_columns = combined_signals.columns.tolist()
    
    # Convert original signals to images
    original_conversion_results = convert_signals_to_images(
        signal_df=combined_signals,
        signal_columns=signal_columns,
        window_size=window_size,
        stride=stride,
        output_dir='original_images',
        save_format=save_format,
        colormap=colormap,
        image_base_dir=image_dir
    )
    
    # Convert scaled signals to images
    scaled_conversion_results = convert_signals_to_images(
        signal_df=scaled_signals,
        signal_columns=signal_columns,
        window_size=window_size,
        stride=stride,
        output_dir='scaled_images',
        save_format=save_format,
        colormap=colormap,
        image_base_dir=image_dir
    )
    
    results['signal_to_image'] = {
        'original_results': original_conversion_results,
        'scaled_results': scaled_conversion_results
    }
    
    # =============================================================================
    # STEP 3: MODEL TRAINING
    # =============================================================================
    print(f"\n{'🤖 STEP 3: MODEL TRAINING'}")
    print("="*80)
    
    # Use original images for training (you can modify this to use scaled or both)
    training_data_dir = original_conversion_results['image_dir']
    
    if training_data_dir and os.path.exists(training_data_dir):
        # Train the model
        trained_model, test_loss, test_acc = train_signet_model(
            data_dir=training_data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_dir=model_dir
        )
        
        results['training'] = {
            'model': trained_model,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'model_dir': model_dir
        }
    else:
        print("❌ No training data directory found. Skipping model training.")
        results['training'] = None
    
    # =============================================================================
    # PIPELINE COMPLETION SUMMARY
    # =============================================================================
    print(f"\n{'🎉 PIPELINE COMPLETION SUMMARY'}")
    print("="*80)
    
    print(f"📁 Output directory: {output_base_dir}")
    print(f"📊 Processed signals: {processed_signal_dir}")
    print(f"🖼️  Encoded images: {image_dir}")
    print(f"🤖 Trained models: {model_dir}")
    
    if results.get('training'):
        print(f"🎯 Final test accuracy: {results['training']['test_accuracy']:.4f}")
    
    print("\n✅ SIGNET pipeline completed successfully!")
    
    return results


# =============================================================================
# EXAMPLE USAGE AND CONFIGURATION
# =============================================================================

# Example configuration for running the complete pipeline
EXAMPLE_CONFIG = {
    # Signal file paths (modify these paths)
    'signal_file_paths': [
        "path/to/morning_signals.csv",
        "path/to/noon_signals.csv", 
        "path/to/evening_signals.csv"
    ],
    
    # Signal preprocessing parameters
    'initial_trim': 1000,
    'target_samples': 25000,
    'iqr_multiplier': 2.0,
    'scaling_sigma': 0.1,
    
    # Signal-to-image parameters
    'window_size': 64,
    'stride': 32,
    'save_format': SaveFormat.PNG,
    'colormap': 'viridis',
    
    # Model training parameters
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    
    # Directory parameters
    'output_base_dir': './signet_output',
    'signal_output_dir': 'processed_signals',
    'image_output_dir': 'encoded_images',
    'model_output_dir': 'trained_models'
}

print("📋 SIGNET Pipeline Configuration loaded!")
print("🔧 Modify the EXAMPLE_CONFIG dictionary with your file paths and parameters")
print("🚀 Run: results = run_complete_pipeline(**EXAMPLE_CONFIG)")

# Uncomment the following line to run the pipeline with example configuration:
# results = run_complete_pipeline(**EXAMPLE_CONFIG)
