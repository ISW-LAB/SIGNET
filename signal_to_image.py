"""
SIGNET Signal-to-Image Conversion Module

This module provides classes and functions for converting time series signals
to image representations using MTF, GAF, and RP encoding methods.

Author: Open Source Community
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm


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
            if window_idx % 100 == 0 and window_idx > 0:
                print(f"  🔢 Processing window {window_idx}/{len(windows)}")
            
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


if __name__ == "__main__":
    print("🖼️ SIGNET Signal-to-Image Module")
    print("Available encoders: MTF_Encoder, GAF_Encoder, RP_Encoder")
    print("Available functions: convert_signals_to_images()")