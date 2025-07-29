#!/usr/bin/env python3
"""
SIGNET Main Pipeline

This script executes the complete SIGNET pipeline using the three separate modules:
- signal_preprocessing.py
- signal_to_image.py  
- model.py

Author: Open Source Community
License: MIT

Usage:
    python main_pipeline.py
"""

import os
import sys
from typing import Dict, Any

# Import SIGNET modules
try:
    from signal_preprocessing import (
        load_signal_data,
        preprocess_signals,
        combine_signal_measurements,
        apply_signal_scaling
    )
    
    from signal_to_image import (
        convert_signals_to_images,
        SaveFormat
    )
    
    from model import (
        train_signet_model,
        test_trained_model
    )
    
    print("✅ All SIGNET modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Error importing SIGNET modules: {e}")
    print("Please ensure the following files are in the same directory:")
    print("- signal_preprocessing.py")
    print("- signal_to_image.py")
    print("- model.py")
    sys.exit(1)


# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

class SIGNETConfig:
    """Configuration class for SIGNET pipeline"""
    
    # 📁 Data file paths (MODIFY THESE PATHS TO YOUR DATA)
    SIGNAL_FILE_PATHS = [
        "data/morning_signals.csv",    # Morning (09h) measurements
        "data/noon_signals.csv",       # Noon (12h) measurements  
        "data/evening_signals.csv"     # Evening (18h) measurements
    ]
    
    # 🔧 Signal preprocessing parameters
    PREPROCESSING = {
        'initial_trim': 1000,        # Remove first N samples for stabilization
        'target_samples': 25000,     # Target length per signal
        'iqr_multiplier': 2.0,       # Outlier detection threshold (IQR multiplier)
        'scaling_sigma': 0.1         # Data augmentation noise level
    }
    
    # 🖼️ Signal-to-image conversion parameters
    IMAGE_CONVERSION = {
        'window_size': 64,           # Sliding window size
        'stride': 32,                # Window stride
        'save_format': SaveFormat.PNG,  # Output format (PNG, NUMPY, or BOTH)
        'colormap': 'viridis',       # Image colormap
        'dpi': 100                   # Image resolution
    }
    
    # 🤖 Model training parameters
    TRAINING = {
        'num_epochs': 50,            # Training epochs
        'batch_size': 32,            # Batch size
        'learning_rate': 0.001,      # Learning rate
        'num_modalities': 6,         # Number of modalities (MTF, GAF, RP × 2 versions)
        'num_classes': 3,            # Number of classes
        'dropout_rate': 0.5          # Dropout rate
    }
    
    # 📁 Output directories
    OUTPUT = {
        'base_dir': './signet_output',
        'signals_dir': 'processed_signals',
        'images_dir': 'encoded_images',
        'models_dir': 'trained_models'
    }


def check_data_files(file_paths):
    """Check if all required data files exist"""
    print("📋 Checking data files...")
    missing_files = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease update the file paths in SIGNETConfig.SIGNAL_FILE_PATHS")
        return False
    
    print("✅ All data files found!")
    return True


def create_output_directories(config):
    """Create necessary output directories"""
    print("📁 Creating output directories...")
    
    base_dir = config.OUTPUT['base_dir']
    signals_dir = os.path.join(base_dir, config.OUTPUT['signals_dir'])
    images_dir = os.path.join(base_dir, config.OUTPUT['images_dir'])
    models_dir = os.path.join(base_dir, config.OUTPUT['models_dir'])
    
    os.makedirs(signals_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"  📂 Base: {base_dir}")
    print(f"  📊 Signals: {signals_dir}")
    print(f"  🖼️ Images: {images_dir}")
    print(f"  🤖 Models: {models_dir}")
    
    return {
        'base_dir': base_dir,
        'signals_dir': signals_dir,
        'images_dir': images_dir,
        'models_dir': models_dir
    }


def run_signal_preprocessing(config, output_dirs):
    """Execute signal preprocessing step"""
    print("\n" + "="*80)
    print("🔧 STEP 1: SIGNAL PREPROCESSING")
    print("="*80)
    
    # Load signal data
    print("\n📁 Loading signal data...")
    morning_signals, noon_signals, evening_signals = load_signal_data(
        config.SIGNAL_FILE_PATHS
    )
    
    # Preprocess signals
    print("\n🔄 Preprocessing signals...")
    morning_processed, noon_processed, evening_processed = preprocess_signals(
        morning_signals, noon_signals, evening_signals,
        **config.PREPROCESSING
    )
    
    # Combine signals
    print("\n🔗 Combining signal measurements...")
    combined_signals = combine_signal_measurements(
        morning_processed, noon_processed, evening_processed
    )
    
    # Apply scaling for data augmentation
    print("\n📈 Applying signal scaling...")
    scaled_signals = apply_signal_scaling(
        combined_signals, 
        sigma=config.PREPROCESSING['scaling_sigma']
    )
    
    # Save processed signals
    original_signal_path = os.path.join(output_dirs['signals_dir'], 'combined_signals.csv')
    scaled_signal_path = os.path.join(output_dirs['signals_dir'], 'scaled_signals.csv')
    
    combined_signals.to_csv(original_signal_path, index=False)
    scaled_signals.to_csv(scaled_signal_path, index=False)
    
    print(f"\n💾 Processed signals saved:")
    print(f"  📊 Original: {original_signal_path}")
    print(f"  📊 Scaled: {scaled_signal_path}")
    
    # Display basic statistics
    print(f"\n📈 Signal Statistics:")
    print(f"  🔢 Combined signals shape: {combined_signals.shape}")
    print(f"  📋 Signal columns: {list(combined_signals.columns)}")
    print(f"  📊 Value range: {combined_signals.min().min():.3f} to {combined_signals.max().max():.3f}")
    
    return {
        'combined_signals': combined_signals,
        'scaled_signals': scaled_signals,
        'original_path': original_signal_path,
        'scaled_path': scaled_signal_path
    }


def run_image_conversion(config, output_dirs, preprocessing_results):
    """Execute signal-to-image conversion step"""
    print("\n" + "="*80)
    print("🖼️ STEP 2: SIGNAL-TO-IMAGE CONVERSION")
    print("="*80)
    
    combined_signals = preprocessing_results['combined_signals']
    scaled_signals = preprocessing_results['scaled_signals']
    
    # Get signal columns
    signal_columns = combined_signals.columns.tolist()
    print(f"\n📊 Processing {len(signal_columns)} signal channels: {signal_columns}")
    
    # Convert original signals to images
    print("\n🔄 Converting original signals to images...")
    original_conversion_results = convert_signals_to_images(
        signal_df=combined_signals,
        signal_columns=signal_columns,
        output_dir='original_images',
        image_base_dir=output_dirs['images_dir'],
        **config.IMAGE_CONVERSION
    )
    
    # Convert scaled signals to images
    print("\n🔄 Converting scaled signals to images...")
    scaled_conversion_results = convert_signals_to_images(
        signal_df=scaled_signals,
        signal_columns=signal_columns,
        output_dir='scaled_images',
        image_base_dir=output_dirs['images_dir'],
        **config.IMAGE_CONVERSION
    )
    
    # Display conversion results
    print(f"\n📈 Conversion Results:")
    print(f"  🖼️ Original images: {original_conversion_results['total_files']} files")
    print(f"  🖼️ Scaled images: {scaled_conversion_results['total_files']} files")
    print(f"  📁 Images directory: {output_dirs['images_dir']}")
    
    return {
        'original_results': original_conversion_results,
        'scaled_results': scaled_conversion_results
    }


def run_model_training(config, output_dirs, conversion_results):
    """Execute model training step"""
    print("\n" + "="*80)
    print("🤖 STEP 3: DEEP LEARNING MODEL TRAINING")
    print("="*80)
    
    # Use original images for training (you can change this to scaled images if preferred)
    training_data_dir = conversion_results['original_results']['image_dir']
    
    print(f"\n📁 Training data directory: {training_data_dir}")
    print(f"🔧 Training configuration: {config.TRAINING}")
    
    if training_data_dir and os.path.exists(training_data_dir):
        # Train the model
        print("\n🚀 Starting model training...")
        trained_model, test_loss, test_accuracy = train_signet_model(
            data_dir=training_data_dir,
            num_epochs=config.TRAINING['num_epochs'],
            batch_size=config.TRAINING['batch_size'],
            learning_rate=config.TRAINING['learning_rate'],
            save_dir=output_dirs['models_dir']
        )
        
        print(f"\n🎯 Final Results:")
        print(f"  📉 Test Loss: {test_loss:.4f}")
        print(f"  🏆 Test Accuracy: {test_accuracy:.4f}")
        print(f"  💾 Model saved in: {output_dirs['models_dir']}")
        
        return {
            'model': trained_model,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'model_dir': output_dirs['models_dir']
        }
    else:
        print(f"\n❌ Training data directory not found: {training_data_dir}")
        print("Please check that Step 2 completed successfully.")
        return None


def run_complete_pipeline():
    """Execute the complete SIGNET pipeline"""
    print("🌟" + "="*79)
    print("🌟 SIGNET: Complete Plant Electrical Signal Classification Pipeline")
    print("🌟" + "="*79)
    
    # Initialize configuration
    config = SIGNETConfig()
    
    # Check data files
    if not check_data_files(config.SIGNAL_FILE_PATHS):
        return None
    
    # Create output directories
    output_dirs = create_output_directories(config)
    
    try:
        # Step 1: Signal preprocessing
        preprocessing_results = run_signal_preprocessing(config, output_dirs)
        print("✅ Step 1 completed successfully!")
        
        # Step 2: Image conversion
        conversion_results = run_image_conversion(config, output_dirs, preprocessing_results)
        print("✅ Step 2 completed successfully!")
        
        # Step 3: Model training
        training_results = run_model_training(config, output_dirs, conversion_results)
        if training_results:
            print("✅ Step 3 completed successfully!")
        else:
            print("⚠️ Step 3 completed with issues!")
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETION SUMMARY")
        print("="*80)
        
        print(f"\n📁 Output Structure:")
        print(f"  📂 Base directory: {output_dirs['base_dir']}")
        print(f"  ├── 📊 Processed signals: {config.OUTPUT['signals_dir']}/")
        print(f"  ├── 🖼️ Encoded images: {config.OUTPUT['images_dir']}/")
        print(f"  └── 🤖 Trained models: {config.OUTPUT['models_dir']}/")
        
        if training_results:
            print(f"\n🎯 Final Performance:")
            print(f"  🏆 Test Accuracy: {training_results['test_accuracy']:.4f}")
            print(f"  📉 Test Loss: {training_results['test_loss']:.4f}")
        
        print("\n✅ SIGNET pipeline completed successfully!")
        
        return {
            'preprocessing': preprocessing_results,
            'conversion': conversion_results,
            'training': training_results,
            'output_dirs': output_dirs
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("Please check the error messages above and fix any issues.")
        return None


def run_individual_step(step_name: str):
    """Run an individual step of the pipeline"""
    config = SIGNETConfig()
    output_dirs = create_output_directories(config)
    
    if step_name.lower() == 'preprocessing':
        print("🔧 Running Signal Preprocessing only...")
        return run_signal_preprocessing(config, output_dirs)
        
    elif step_name.lower() == 'conversion':
        print("🖼️ Running Image Conversion only...")
        # Need to load preprocessing results
        try:
            import pandas as pd
            combined_path = os.path.join(output_dirs['signals_dir'], 'combined_signals.csv')
            scaled_path = os.path.join(output_dirs['signals_dir'], 'scaled_signals.csv')
            
            if os.path.exists(combined_path) and os.path.exists(scaled_path):
                combined_signals = pd.read_csv(combined_path)
                scaled_signals = pd.read_csv(scaled_path)
                
                preprocessing_results = {
                    'combined_signals': combined_signals,
                    'scaled_signals': scaled_signals
                }
                
                return run_image_conversion(config, output_dirs, preprocessing_results)
            else:
                print("❌ Preprocessing results not found. Run preprocessing step first.")
                return None
        except Exception as e:
            print(f"❌ Error loading preprocessing results: {e}")
            return None
            
    elif step_name.lower() == 'training':
        print("🤖 Running Model Training only...")
        # Need to check if images exist
        images_dir = os.path.join(output_dirs['images_dir'], 'original_images')
        if os.path.exists(images_dir):
            conversion_results = {
                'original_results': {'image_dir': images_dir},
                'scaled_results': {'image_dir': os.path.join(output_dirs['images_dir'], 'scaled_images')}
            }
            return run_model_training(config, output_dirs, conversion_results)
        else:
            print("❌ Image conversion results not found. Run conversion step first.")
            return None
    else:
        print(f"❌ Unknown step: {step_name}")
        print("Available steps: preprocessing, conversion, training")
        return None


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='SIGNET Plant Signal Classification Pipeline')
    parser.add_argument('--step', type=str, choices=['preprocessing', 'conversion', 'training'], 
                       help='Run only a specific step')
    parser.add_argument('--config', action='store_true', 
                       help='Print current configuration and exit')
    
    args = parser.parse_args()
    
    # Print configuration if requested
    if args.config:
        config = SIGNETConfig()
        print("📋 Current SIGNET Configuration:")
        print("="*50)
        print(f"Signal files: {config.SIGNAL_FILE_PATHS}")
        print(f"Preprocessing: {config.PREPROCESSING}")
        print(f"Image conversion: {config.IMAGE_CONVERSION}")
        print(f"Training: {config.TRAINING}")
        print(f"Output: {config.OUTPUT}")
        sys.exit(0)
    
    # Run specific step or complete pipeline
    if args.step:
        results = run_individual_step(args.step)
    else:
        results = run_complete_pipeline()
    
    # Exit with appropriate code
    if results is not None:
        print("\n🎉 Execution completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Execution failed!")
        sys.exit(1)