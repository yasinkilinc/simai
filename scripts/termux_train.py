#!/usr/bin/env python3
"""
Termux Training Script - Minimal PyTorch Training for Mobile
Bu scripti Xiaomi 14T Pro'da Termux ile çalıştırın.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import time
from pathlib import Path
import os

# ====================================
# Simple CNN Model
# ====================================
class MiniPhysiognomyNet(nn.Module):
    """Lightweight model for mobile training"""
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ====================================
# Dataset
# ====================================
class MobileDataset(Dataset):
    """Simple dataset for mobile training"""
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform or self.default_transform()
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.img_dir / row['filename']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Label
        label = int(row['target_face_shape'])
        
        return img, label


# ====================================
# Training Loop
# ====================================
def train_mobile(data_dir='mobile_dataset', epochs=5, batch_size=4, lr=1e-3):
    """
    Main training function for Termux
    
    Args:
        data_dir: Path to extracted dataset (must contain data.csv and images/)
        epochs: Number of training epochs
        batch_size: Batch size (keep small for mobile)
        lr: Learning rate
    """
    print("=" * 50)
    print("🚀 Termux Mobile Training - Başlatılıyor")
    print("=" * 50)
    
    # Device
    device = 'cpu'  # Termux only supports CPU
    print(f"Device: {device}")
    
    # Dataset
    print(f"\n📂 Dataset yükleniyor: {data_dir}")
    dataset = MobileDataset(
        csv_path=os.path.join(data_dir, 'data.csv'),
        img_dir=os.path.join(data_dir, 'images')
    )
    print(f"✅ {len(dataset)} resim yüklendi")
    
    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Batch size: {batch_size}, Batches: {len(loader)}")
    
    # Model
    print("\n🤖 Model oluşturuluyor...")
    num_classes = int(dataset.df['target_face_shape'].max()) + 1
    model = MiniPhysiognomyNet(num_classes=num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametreleri: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    print("\n" + "=" * 50)
    print(f"🔥 Eğitim başlıyor - {epochs} epoch")
    print("=" * 50)
    
    total_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress
            if batch_idx % 5 == 0 or batch_idx == len(loader) - 1:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Time: {epoch_time:.1f}s")
        print("-" * 50)
    
    # Total time
    total_time = time.time() - total_start
    print(f"\n✅ Eğitim tamamlandı!")
    print(f"Toplam süre: {total_time:.1f}s ({total_time/60:.2f} dk)")
    print(f"Epoch başına: {total_time/epochs:.1f}s")
    
    # Save model
    checkpoint_path = 'mobile_model_checkpoint.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'loss': avg_loss,
        'accuracy': accuracy
    }, checkpoint_path)
    print(f"\n💾 Model kaydedildi: {checkpoint_path}")
    
    return model

# ====================================
# Thermal Monitoring (Optional)
# ====================================
def print_device_stats():
    """Print device statistics (Termux only)"""
    try:
        import subprocess
        result = subprocess.run(['termux-battery-status'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\n🌡️ Device Stats:")
            print(result.stdout)
    except:
        pass


# ====================================
# Main
# ====================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Termux Mobile Training')
    parser.add_argument('--data_dir', type=str, default='mobile_dataset',
                       help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Print device stats
    print_device_stats()
    
    # Train
    model = train_mobile(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    print("\n🎉 Script tamamlandı!")
