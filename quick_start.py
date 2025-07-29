#!/usr/bin/env python3
"""
SIGNET Quick Start Script

A simplified interface for running the SIGNET pipeline with minimal configuration.

Usage:
    python quick_start.py
"""

from main_pipeline import SIGNETConfig, run_complete_pipeline
import os

def quick_setup():
    """Quick setup with minimal configuration"""
    print("🚀 SIGNET Quick Start")
    print("="*50)
    
    # Quick configuration - modify these paths for your data
    config = SIGNETConfig()
    
    # Example data paths - update these to your actual file locations
    config.SIGNAL_FILE_PATHS = [
        "data/morning_signals.csv",
        "data/noon_signals.csv", 
        "data/evening_signals.csv"
    ]
    
    # Quick settings for faster testing
    config.PREPROCESSING['target_samples'] = 10000  # Smaller for testing
    config.IMAGE_CONVERSION['window_size'] = 32     # Smaller for testing
    config.TRAINING['num_epochs'] = 10              # Fewer epochs for testing
    config.TRAINING['batch_size'] = 16              # Smaller batch size
    
    return config

def main():
    """Main execution function"""
    print("🌱 Welcome to SIGNET!")
    print("This script will run the complete pipeline with default settings.")
    print("\n📋 Before starting, please ensure:")
    print("1. You have your signal CSV files ready")
    print("2. The file paths in this script match your data location")
    print("3. You have sufficient disk space for output files")
    
    # Ask user confirmation
    response = input("\nDo you want to continue? (y/n): ").lower().strip()
    if response != 'y' and response != 'yes':
        print("👋 Goodbye!")
        return
    
    # Setup configuration
    config = quick_setup()
    
    # Check if data files exist
    missing_files = []
    for file_path in config.SIGNAL_FILE_PATHS:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n💡 Please update the file paths in quick_start.py or create sample data files.")
        return
    
    print(f"\n✅ All data files found!")
    print(f"📊 Configuration: {config.PREPROCESSING['target_samples']} samples, "
          f"{config.IMAGE_CONVERSION['window_size']} window size, "
          f"{config.TRAINING['num_epochs']} epochs")
    
    # Run the pipeline
    print("\n🚀 Starting SIGNET pipeline...")
    results = run_complete_pipeline()
    
    if results:
        print("\n🎉 Quick start completed successfully!")
        if results.get('training') and results['training']:
            accuracy = results['training']['test_accuracy']
            print(f"🎯 Final accuracy: {accuracy:.2%}")
    else:
        print("\n❌ Quick start failed. Please check the error messages above.")

if __name__ == "__main__":
    main()