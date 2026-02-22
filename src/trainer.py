import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import os
from datetime import datetime
from src.loss import ArcFaceLoss

class Trainer:
    """
    Trainer class for PhysiognomyNet.
    Handles training loop, validation, checkpointing, and logging.
    """
    def __init__(self, model, device=None, learning_rate=0.001, num_classes=7):
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.model = model.to(self.device)
        
        # Define loss functions
        # ArcFace for Face Shape
        self.criterion_arcface = ArcFaceLoss(in_features=512, out_features=num_classes).to(self.device)
        self.criterion_reg = nn.MSELoss()
        
        # Optimizer
        # Optimize model parameters AND ArcFace parameters
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        params_to_update += [p for p in self.criterion_arcface.parameters()]
        
        self.optimizer = optim.Adam(params_to_update, lr=learning_rate)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
    def train_model(self, dataloaders, dataset_sizes, num_epochs=25, checkpoint_dir='models/checkpoints', gradient_accumulation_steps=1):
        """
        Main training loop.
        """
        since = time.time()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)
            
            # Calculate total steps for progress bar
            total_train_batches = len(dataloaders['train'])
            total_steps = num_epochs * total_train_batches
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    self.criterion_arcface.train()
                    print(f"--> [FAZ: EĞİTİM] Epoch {epoch+1} için eğitim başladı...")
                else:
                    self.model.eval()
                    self.criterion_arcface.eval()
                    print(f"--> [FAZ: DOĞRULAMA] Model performansı test ediliyor...")
                    
                running_loss = 0.0
                running_corrects = 0 # For accuracy
                total_samples = 0
                
                # Iterate over data.
                for batch_idx, (images, landmarks, targets) in enumerate(dataloaders[phase]):
                    images = images.to(self.device)
                    landmarks = landmarks.to(self.device)
                    
                    # Move targets to device
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images, landmarks)
                        
                        # Calculate total loss
                        loss = 0.0
                        
                        # 1. ArcFace Loss for Face Shape
                        if 'target_face_shape' in targets:
                            embedding = outputs['embedding']
                            label = targets['target_face_shape']
                            # ArcFace Loss
                            loss += self.criterion_arcface(embedding, label)
                            
                            # Calculate Accuracy
                            # For ArcFace, we use cosine similarity to class centers (in criterion_arcface.weight)
                            # Or simply use the output of criterion_arcface.forward (which returns logits/cosine)
                            # Wait, criterion_arcface forward returns LOSS, not logits if label is provided.
                            # We need logits to calculate accuracy.
                            # Let's call forward with label=None to get cosine logits
                            with torch.no_grad():
                                logits = self.criterion_arcface(embedding, label=None)
                                _, preds = torch.max(logits, 1)
                                running_corrects += torch.sum(preds == label.data)
                                total_samples += label.size(0)
                        
                        # 2. Regression Losses
                        for trait_name, output in outputs.items():
                            if trait_name == 'embedding': continue
                            
                            if trait_name in targets:
                                target = targets[trait_name]
                                loss += self.criterion_reg(output, target.unsqueeze(1))
                        
                        # Normalize loss for gradient accumulation
                        if phase == 'train':
                            loss = loss / gradient_accumulation_steps
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            
                            # Gradient accumulation: only step every N batches
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                            
                    # Statistics
                    running_loss += loss.item() * images.size(0)
                    
                    # Batch-level progress for UI (only for training phase)
                    if phase == 'train' and batch_idx % 5 == 0:
                        current_step = epoch * total_train_batches + batch_idx
                        print(f"[PROGRESS] {current_step}/{total_steps}", flush=True)
                        # Optional: Print batch loss
                        # print(f"   Batch {batch_idx}/{total_train_batches} Loss: {loss.item():.4f}")
                    
                if phase == 'train':
                    self.scheduler.step()
                    
                epoch_loss = running_loss / dataset_sizes[phase]
                history[f'{phase}_loss'].append(epoch_loss)
                
                # Calculate Epoch Accuracy
                epoch_acc = 0.0
                if total_samples > 0:
                    epoch_acc = running_corrects.float() / total_samples
                    print(f"   📊 {phase.upper()} Doğruluk (Accuracy): %{epoch_acc*100:.2f}")
                
                print(f"   📉 {phase.upper()} Kayıp (Loss): {epoch_loss:.4f}")
                
                # Deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print(f"   ⭐ Yeni en iyi model! (Eski Kayıp: {best_loss:.4f} -> Yeni: {epoch_loss:.4f})")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_arcface_wts = copy.deepcopy(self.criterion_arcface.state_dict())
                    
                    # Save checkpoint
                    ckpt_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}_loss_{epoch_loss:.4f}.pth')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'arcface_state_dict': self.criterion_arcface.state_dict(),
                        'epoch': epoch,
                        'loss': epoch_loss,
                        'accuracy': epoch_acc
                    }, ckpt_path)
                    
            # Print progress tag for UI (End of epoch)
            current_step = (epoch + 1) * total_train_batches
            print(f"[PROGRESS] {current_step}/{total_steps}", flush=True)
            print()
            
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        self.criterion_arcface.load_state_dict(best_arcface_wts)
        
        return self.model, self.criterion_arcface, history
