# 🌱 SIGNET

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

This study presents a signal-to-image classification approach for analyzing plant bioelectrical responses under varying irrigation conditions. The proposed framework, SIGNET aims to enhance physiological interpretation and multi-class prediction performance in controlled environment agriculture.

## ✨ Features

- **Signal Preprocessing**: Automated outlier detection and cleaning
- **Multi-Modal Encoding**: MTF, GAF, RP time-series to image conversion
- **Deep Learning**: Multi-modal CNN based backbone
- **One-Line Execution**: Complete pipeline from raw signals to trained model

## 🚀 Quick Start

### Installation
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn Pillow tqdm
git clone https://github.com/ISW-LAB/SIGNET.git
cd SIGNET
```

### Usage
```python
from signet_pipeline import run_complete_pipeline

config = {
    'signal_file_paths': [
        "data/morning_signals.csv",
        "data/noon_signals.csv", 
        "data/evening_signals.csv"
    ],
    'output_base_dir': './signet_output',
    'num_epochs': 50,
    'batch_size': 32
}

# Run complete pipeline
results = run_complete_pipeline(**config)
print(f"Final Accuracy: {results['training']['test_accuracy']:.4f}")
```

## 📊 Pipeline Overview

```
Raw Signals → Preprocessing → MTF/GAF/RP Encoding → Multi-Modal CNN → Classification
```

### 3-Stage Process
1. **Signal Preprocessing**: Clean and normalize electrical signals
2. **Image Encoding**: Convert time series to visual representations
3. **Model Training**: Multi-modal CNN classification (3 irrigation classes)

## ⚙️ Configuration

### Key Parameters
```python
{
    # Data paths
    'signal_file_paths': ["morning.csv", "noon.csv", "evening.csv"],
    
    # Signal processing
    'window_size': 64,           # Sliding window size
    'stride': 32,                # Window stride
    'initial_trim': 1000,        # Remove unstable samples
    
    # Training
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001
}
```

### Input Data Format
```csv
timestamp,DC_Voltage_CH103,DC_Voltage_CH104,DC_Voltage_CH105
0,2.45,2.37,2.41
1,2.46,2.38,2.42
...
```

## 📁 Output Structure

```
signet_output/
├── processed_signals/       # Cleaned signal data
├── encoded_images/          # MTF/GAF/RP images  
└── trained_models/          # Best model checkpoint
```

## 🔧 Individual Module Usage

### Signal Preprocessing Only
```python
from signet_pipeline import preprocess_signals

morning_clean, noon_clean, evening_clean = preprocess_signals(
    morning_data, noon_data, evening_data
)
```

### Image Encoding Only
```python
from signet_pipeline import convert_signals_to_images

results = convert_signals_to_images(
    signal_df=your_signals,
    signal_columns=['CH-103', 'CH-104'],
    window_size=64
)
```

### Model Training Only
```python
from signet_pipeline import train_signet_model

model, test_loss, test_acc = train_signet_model(
    data_dir='path/to/images',
    num_epochs=50
)
```

## 🏗️ Model Architecture

- **Input**: 6 modalities (MTF, GAF, RP × original/scaled)
- **Backbone**: MobileNet V2 (ImageNet pretrained)
- **Fusion**: Late concatenation + 3-layer MLP
- **Output**: 3-class classification (irrigated_1/2/3)

## 📈 Performance

- **Accuracy**: >90% on test set
- **Training Time**: ~30 minutes (50 epochs, GPU)
- **Memory**: ~4GB GPU memory

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.


## 📞 Contact Us

  If you have any questions or provide your cell images, please contact us by email
  Hongseok Oh: hs.oh-isw@cbnu.ac.kr
  Yeongyu Lee: yg.lee-isw@chungbuk.ac.kr
