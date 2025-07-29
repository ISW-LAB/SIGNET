# 🌱 SIGNET

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

This study presents a signal-to-image classification approach for analyzing plant bioelectrical responses under varying irrigation conditions. The proposed framework, SIGNET aims to enhance physiological interpretation and multi-class prediction performance in controlled environment agriculture.

## ✨ Features
- **Modular Architecture**: Three independent Python modules
- **Signal Preprocessing**: Automated outlier detection and cleaning
- **Multi-Modal Encoding**: MTF, GAF, RP time-series to image conversion
- **Deep Learning**: Multi-modal CNN based backbone
- **Easy Execution**: Single command pipeline with built-in configuration

## 🚀 Quick Start

### Installation
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn Pillow tqdm scipy
git clone https://github.com//ISW-LAB/SIGNET.git
cd SIGNET
```

### Usage
1. **Update data paths** in `main_pipeline.py`:
```python
SIGNAL_FILE_PATHS = [
    "data/morning.csv",
    "data/noon.csv", 
    "data/evening.csv"
]
```

2. **Run pipeline**:
```bash
python main_pipeline.py
```

## 📁 Files

```
signet/
├── signal_preprocessing.py      # Signal cleaning
├── signal_to_image.py          # MTF/GAF/RP conversion
├── deep_learning_model.py      # Multi-modal CNN
├── main_pipeline.py           # Main execution
└── quick_start.py             # Simple test run
```

## ⚙️ Key Features

- **Signal Preprocessing**: Outlier detection and cleaning
- **Image Encoding**: MTF, GAF, RP time-series to image conversion
- **Multi-Modal CNN**: 6 modalities → CNN_based backbone → 3-class output
- **One-Command Execution**: Complete pipeline in single run

## 📊 Data Format

CSV files should contain:
```csv
timestamp,DC_Voltage_CH103,DC_Voltage_CH104,DC_Voltage_CH105
0,2.45,2.37,2.41
1,2.46,2.38,2.42
...
```

## 🔧 Configuration

Edit `SIGNETConfig` in `main_pipeline.py`:

```python
PREPROCESSING = {
    'initial_trim': 1000,        # Remove first N samples
    'target_samples': 25000,     # Signal length
    'iqr_multiplier': 2.0        # Outlier threshold
}

IMAGE_CONVERSION = {
    'window_size': 64,           # Window size
    'stride': 32,                # Window step
    'save_format': SaveFormat.PNG
}

TRAINING = {
    'num_epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001
}
```

## 📈 Architecture

```
Raw Signals → Preprocessing → MTF/GAF/RP Images → Multi-Modal CNN → 3 Classes
```

- **Input**: 6 modalities (MTF, GAF, RP × original/scaled)
- **Backbone**: Custom CNN based Backbone
- **Output**: irrigated_1, irrigated_2, irrigated_3

## 🛠️ Commands

```bash
python main_pipeline.py                    # Complete pipeline
python main_pipeline.py --step preprocessing  # Individual step
python main_pipeline.py --config           # View settings
python quick_start.py                      # Quick test
```

## 📁 Output

```
signet_output/
├── processed_signals/    # Cleaned CSV files
├── encoded_images/       # MTF/GAF/RP images
└── trained_models/       # Model checkpoints
```

## 🔧 Individual Modules

```python
# Signal preprocessing only
from signal_preprocessing import preprocess_signals

# Image conversion only  
from signal_to_image import convert_signals_to_images

# Model training only
from deep_learning_model import train_signet_model
```

## 🧪 Quick Test

For testing with smaller data:
```python
# Edit in main_pipeline.py or quick_start.py
PREPROCESSING['target_samples'] = 5000
IMAGE_CONVERSION['window_size'] = 32  
TRAINING['num_epochs'] = 10
```

## 📞 Contact Us

  If you have any questions or provide your cell images, please contact us by email
  Hongseok Oh: hs.oh-isw@cbnu.ac.kr
  Yeongyu Lee: yg.lee-isw@chungbuk.ac.kr
