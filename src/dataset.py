import os
import cv2
import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    """
    PyTorch Dataset for Face Physiognomy.
    Loads images, landmarks, and corresponding labels/features from CSV.
    """
    def __init__(self, csv_file, root_dir, landmark_file=None, transform=None, target_columns=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the images.
            landmark_file (str, optional): Path to the JSON file with landmarks.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_columns (list, optional): List of column names to be used as targets.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_columns = target_columns if target_columns else ['target_face_shape']
        
        # Load landmarks if provided
        self.landmarks_data = {}
        if landmark_file and os.path.exists(landmark_file):
            with open(landmark_file, 'r') as f:
                self.landmarks_data = json.load(f)
        
        # Filter out rows where image doesn't exist
        self.data_frame = self.data_frame[self.data_frame['image_name'].apply(
            lambda x: os.path.exists(os.path.join(root_dir, x))
        )]
        
        # Create label encoders for categorical targets
        self.label_mappings = {}
        for col in self.target_columns:
            if self.data_frame[col].dtype == 'object':
                unique_vals = sorted(self.data_frame[col].dropna().unique())
                self.label_mappings[col] = {val: i for i, val in enumerate(unique_vals)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        img_name = row['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        
        # 1. Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Landmarks
        # Default to zeros if not found
        landmarks = np.zeros((478, 3), dtype=np.float32)
        if img_name in self.landmarks_data:
            landmarks = np.array(self.landmarks_data[img_name], dtype=np.float32)
            
        # Flatten landmarks: [478, 3] -> [1434]
        # Or keep as [478, 3] depending on model. Let's flatten for MLP.
        # We only use x, y for simplicity or x, y, z. Let's use x, y.
        # landmarks_flat = landmarks[:, :2].flatten() # [956]
        landmarks_flat = torch.tensor(landmarks[:, :2].flatten(), dtype=torch.float32)

        # 3. Get Targets
        targets = {}
        for col in self.target_columns:
            val = row[col]
            
            # Handle categorical
            if col in self.label_mappings:
                if pd.isna(val):
                    target_val = -1 
                else:
                    target_val = self.label_mappings[col][val]
                targets[col] = torch.tensor(target_val, dtype=torch.long)
            else:
                # Continuous value
                if pd.isna(val):
                    val = 0.0 
                targets[col] = torch.tensor(val, dtype=torch.float32)

        # 4. Apply Transforms
        if self.transform:
            image = self.transform(image)
        else:
            default_tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_tf(image)

        return image, landmarks_flat, targets

def get_transforms(train=True):
    """
    Get data augmentation transforms.
    """
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)), # EfficientNet-B4 input size is usually larger (380), but 300 is okay
            transforms.RandomCrop(256), # Let's target 256 or 380 depending on compute
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
