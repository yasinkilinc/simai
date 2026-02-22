import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from src.dataset import FaceDataset, get_transforms
from src.models import PhysiognomyNet
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning Physiognomy Model (EfficientNet + ArcFace + Landmarks)')
    parser.add_argument('--data_csv', type=str, default='dataset/export/data.csv', help='Path to CSV file')
    parser.add_argument('--img_dir', type=str, default='dataset/export/images', help='Path to images directory')
    parser.add_argument('--landmark_file', type=str, default='dataset/export/landmarks.json', help='Path to landmarks JSON file')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    parser.add_argument('--output_path', type=str, default='models/physiognomy_net_efficientnet_arcface.pth', help='Path to save the model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_csv):
        print(f"Error: Data CSV not found at {args.data_csv}")
        return
        
    print(f"Starting training with EfficientNet-B4 + ArcFace + Landmarks", flush=True)
    
    # 1. Define Targets
    print("--> Hedef değişkenler tanımlanıyor...", flush=True)
    target_columns = ['target_face_shape'] 
    # Add more targets if they exist in CSV
    # target_columns.extend(['target_forehead_width', 'target_nose_width'])
    
    # 2. Create Dataset
    print("--> Veri seti yükleniyor (Resimler ve Landmarklar)...", flush=True)
    full_dataset = FaceDataset(
        csv_file=args.data_csv,
        root_dir=args.img_dir,
        landmark_file=args.landmark_file,
        transform=get_transforms(train=True),
        target_columns=target_columns
    )
    
    # 3. Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {'train': train_size, 'val': val_size}
    print(f"Dataset sizes: {dataset_sizes}")
    
    # 4. Define Model
    print("--> Model oluşturuluyor (Pretrained ağırlıklar indirilebilir)...", flush=True)
    # Get number of classes for face_shape from dataset
    num_classes = len(full_dataset.label_mappings.get('target_face_shape', {}))
    if num_classes == 0: num_classes = 7 # Default fallback
    
    num_traits = {
        'target_face_shape': num_classes
        # Add other heads here matching target_columns
    }
    
    model = PhysiognomyNet(num_traits=num_traits)
    
    # 5. Train
    print("--> Eğitim döngüsü başlatılıyor...", flush=True)
    trainer = Trainer(model, learning_rate=args.lr, num_classes=num_classes)
    model, arcface, history = trainer.train_model(dataloaders, dataset_sizes, num_epochs=args.epochs, gradient_accumulation_steps=args.accumulation_steps)
    
    # 6. Save Final Model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_path = args.output_path
    
    # Save everything needed for inference
    torch.save({
        'model_state_dict': model.state_dict(),
        'arcface_state_dict': arcface.state_dict(),
        'label_mappings': full_dataset.label_mappings,
        'num_traits': num_traits
    }, save_path)
    
    print(f"Final model saved to {save_path}")
    
    # 7. Save Metadata
    import json
    from datetime import datetime
    
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0.0
    
    metadata = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'backbone': 'efficientnet_b4', # Hardcoded for now as per script logic
        'epochs': args.epochs,
        'final_train_loss': float(f"{final_train_loss:.4f}"),
        'final_val_loss': float(f"{final_val_loss:.4f}"),
        'num_classes': num_classes,
        'targets': target_columns
    }
    
    metadata_path = save_path.replace('.pth', '.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
