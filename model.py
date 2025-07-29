"""
SIGNET Deep Learning Model Module

This module provides the multi-modal CNN architecture and training functions
for plant electrical signal 

Author: Open Source Community
License: MIT
"""
    
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
import os
import numpy as np

class SIGNET_Dataset(Dataset):
    """Dataset for loading MTF, GAF, RP images"""
    def __init__(self, root_dir, transform=None, target_subclasses=['MTF', 'GAF', 'RP', 'MTF_scaling', 'GAF_scaling', 'RP_scaling']):
        self.root_dir = root_dir
        self.transform = transform
        self.target_subclasses = target_subclasses
        self.data = []
        self.class_to_idx = {'irrigated_1': 0, 'irrigated_2': 1, 'irrigated_3': 2}
        self.idx_to_class = {0: 'irrigated_1', 1: 'irrigated_2', 2: 'irrigated_3'}
        
        self._load_data()
    
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
    """Inverted Residual Block"""
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
    """Backbone for feature extraction"""
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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


if __name__ == "__main__":
    print("🤖 SIGNET Deep Learning Model Module")
    print("Available classes: SIGNET_Dataset, SIGNET_Model")
    print("Available functions: train_signet_model(), test_trained_model()")
