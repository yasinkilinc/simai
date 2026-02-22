import os
import sys
import torch
import cv2
import json
import argparse
import numpy as np
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())

from src.models import PhysiognomyNet
import mediapipe as mp

def get_landmarks(image_path):
    """Extract 478 landmarks using MediaPipe"""
    mp_face_mesh = mp.solutions.face_mesh
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        # Convert to list of [x, y]
        lm_list = [[lm.x, lm.y] for lm in landmarks.landmark]
        return np.array(lm_list, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description='Predict using PhysiognomyNet')
    parser.add_argument('--model', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    
    args = parser.parse_args()
    
    # 1. Load Metadata
    # Assume metadata json is same name as model pth
    meta_path = args.model.replace('.pth', '.json')
    if not os.path.exists(meta_path):
        print(json.dumps({'error': 'Metadata file not found'}))
        return
        
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    # 2. Load Model Checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    
    # 3. Initialize Model
    # We need to reconstruct the model with same num_traits
    # Ideally this should be in metadata or checkpoint
    num_traits = checkpoint.get('num_traits')
    label_mappings = checkpoint.get('label_mappings', {})
    
    if not num_traits:
        # Fallback or error
        print(json.dumps({'error': 'Model checkpoint missing configuration'}))
        return
        
    model = PhysiognomyNet(num_traits=num_traits)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Prepare Input
    # Image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = cv2.imread(args.image)
    if img is None:
        print(json.dumps({'error': 'Image could not be read'}))
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0) # [1, C, H, W]
    
    # Landmarks
    landmarks = get_landmarks(args.image)
    if landmarks is None:
        # Fallback: Zeros? Or error?
        # Let's use zeros for robustness but warn
        landmarks = np.zeros((478, 2), dtype=np.float32)
        message = "Yüz algılanamadı, landmarklar sıfırlandı."
    else:
        message = "Yüz başarıyla algılandı."
        
    lm_tensor = torch.tensor(landmarks.flatten(), dtype=torch.float32).unsqueeze(0) # [1, 956]
    
    # 5. Predict
    with torch.no_grad():
        outputs = model(img_tensor, lm_tensor)
        
    # 6. Format Output
    result = {'message': message}
    
    # Embedding size
    result['embedding_size'] = outputs['embedding'].shape[1]
    
    # Face Shape (Classification)
    if 'target_face_shape' in label_mappings:
        # ArcFace Logic:
        # 1. Get embedding from model output
        embedding = outputs['embedding'] # [1, 512]
        
        # 2. Load ArcFace weights (Class Prototypes) from checkpoint
        if 'arcface_state_dict' in checkpoint:
            af_weights = checkpoint['arcface_state_dict']['weight'] # [NumClasses, 512]
            
            # 3. Normalize both
            import torch.nn.functional as F
            embedding_norm = F.normalize(embedding)
            weights_norm = F.normalize(af_weights)
            
            # 4. Calculate Cosine Similarity (Dot Product)
            # [1, 512] x [512, NumClasses] = [1, NumClasses]
            logits = torch.mm(embedding_norm, weights_norm.t())
            
            # 5. Get Prediction
            conf, pred_idx = torch.max(logits, 1)
            pred_idx = pred_idx.item()
            confidence = conf.item()
            
            # 6. Map to Label
            # label_mappings['target_face_shape'] is dict {label: index}
            # We need {index: label}
            idx_to_label = {v: k for k, v in label_mappings['target_face_shape'].items()}
            predicted_label = idx_to_label.get(pred_idx, "Bilinmiyor")
            
            result['face_shape'] = {
                'prediction': predicted_label,
                'confidence': confidence
            }
        else:
            result['face_shape'] = "Model eğitilirken ArcFace ağırlıkları kaydedilmemiş."
    
    # Regression Outputs
    for k, v in outputs.items():
        if k == 'embedding': continue
        # Convert tensor to float
        val = v.item()
        result[k] = val
        
    print(json.dumps(result))

if __name__ == "__main__":
    main()
