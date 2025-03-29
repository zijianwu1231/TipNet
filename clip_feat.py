import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.visual

    def forward(self, x):
        with torch.no_grad():
            # convert x to half precision
            x = x.half()
            # Initial patch embeddings
            x = self.model.conv1(x)  # [B, C, H, W] -> Patch embeddings
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

            x = self.model.ln_pre(x)
            B, H, W, C = x.shape
            x = x.view(B, H*W, C)

            # Pass through transformer blocks
            x = self.model.transformer(x) # [B, N, C]

            # Feature map before global pooling: BNC -> BCHW
            feature_map = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            
        return feature_map

# Load CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess image
image_path = "/bigdata/SurgPose/tooltip/l_mask/frame072.png"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Define the text description
text_description = ["left gripper tips of the surgical instrument", "right gripper tips of the surgical instrument"]  # Example classes
text_tokens = clip.tokenize(text_description).to(device)

# Get text features
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)  # [num_text, 512]
    print(text_features.shape)

breakpoint()

# Initialize feature extractor
feature_extractor = FeatureExtractor(clip_model).to(device)

# Get the intermediate feature map
feature_map = feature_extractor(image)  # [1, C, H, W]
print("Feature map shape:", feature_map.shape)  # [1, C, H, W]

breakpoint()

## Project text features to the feature map

# Normalize text features
text_features /= text_features.norm(dim=-1, keepdim=True)

# Flatten the feature map for similarity calculation: BCHW -> B(HW)C
B, C, H, W = feature_map.shape
feature_map_flat = feature_map.view(B, C, -1)  # [B, C, H*W]
breakpoint()

# Compute similarity between text and each spatial location
breakpoint()
similarity_map = torch.matmul(feature_map_flat, text_features.T)  # [H*W, num_text]
breakpoint()
similarity_map = similarity_map.view(H, W, len(text_descriptions)).cpu().numpy()
