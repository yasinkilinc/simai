import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class PhysiognomyNet(nn.Module):
    """
    Multi-modal Physiognomy Model.
    - Image Branch: EfficientNet-B4
    - Landmark Branch: MLP
    - Fusion: Concatenation -> Embedding
    - Heads: ArcFace (Face Shape) + Regression Heads
    """
    def __init__(self, num_traits=None, landmark_dim=478*2, embedding_dim=512):
        """
        Args:
            num_traits (dict): Dictionary defining output heads. 
            landmark_dim (int): Dimension of flattened landmarks (478 points * 2 coords = 956).
            embedding_dim (int): Size of the shared embedding vector.
        """
        super(PhysiognomyNet, self).__init__()
        
        # 1. Image Branch (EfficientNet-B4)
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        base_model = efficientnet_b4(weights=weights)
        self.img_feature_dim = 1792
        # Remove classifier
        self.img_encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        # 2. Landmark Branch (MLP)
        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.landmark_feature_dim = 256
        
        # 3. Fusion & Embedding
        fusion_dim = self.img_feature_dim + self.landmark_feature_dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim) 
            # No ReLU here, we want raw embedding features. 
            # L2 Normalization is usually done in Loss or explicitly here.
        )
        
        # 4. Heads
        if num_traits is None:
            num_traits = {
                'target_face_shape': 7, 
                'target_forehead_width': 1,
                # ... other defaults
            }
            
        self.heads = nn.ModuleDict()
        self.num_traits = num_traits
        
        for trait_name, output_dim in num_traits.items():
            if trait_name == 'target_face_shape':
                # For ArcFace, we don't need a classification layer here if we use the embedding directly.
                # However, usually we have a linear layer to project to class logits if we are NOT using ArcFace during inference.
                # But for ArcFace training, the Loss function handles the weights.
                # Let's return the embedding itself for this head, or a projection.
                # Strategy: We will return the embedding vector for 'face_shape' key, 
                # and the Trainer will feed it to ArcFaceLoss.
                pass 
            else:
                # Regression Heads
                self.heads[trait_name] = nn.Sequential(
                    nn.Linear(embedding_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, output_dim)
                )
            
    def forward(self, image, landmarks):
        # Image Features
        # efficientnet output is [B, 1792, 1, 1] -> flatten
        img_features = self.img_encoder(image)
        img_features = torch.flatten(img_features, 1) # [B, 1792]
        
        # Landmark Features
        lm_features = self.landmark_encoder(landmarks) # [B, 256]
        
        # Fusion
        combined = torch.cat((img_features, lm_features), dim=1) # [B, 2048]
        
        # Embedding
        embedding = self.embedding_layer(combined) # [B, 512]
        
        # Heads
        outputs = {}
        
        # For Face Shape (Classification), we return the embedding.
        # The ArcFaceLoss will take this embedding and the label.
        # During inference, we can use cosine similarity or train a simple classifier on top.
        # To make it compatible with standard training loop, let's return embedding.
        outputs['embedding'] = embedding
        
        # Regression Outputs
        for trait_name, head in self.heads.items():
            outputs[trait_name] = head(embedding)
            
        return outputs

    def freeze_backbone(self):
        """Freeze image backbone weights"""
        for param in self.img_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze image backbone weights"""
        for param in self.img_encoder.parameters():
            param.requires_grad = True
