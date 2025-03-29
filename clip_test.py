import torch
import clip
from PIL import Image

# def get_attention_maps(model, image):
#     with torch.no_grad():
#         _ = model.encode_image(image)  # Forward pass
#         breakpoint()
#         attn_maps = model.visual.transformer.resblocks[-1].attn.attn_map
#     return attn_maps

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

            breakpoint()

            attn_map = self.model.transformer.resblocks[-1].attn.attn_output_weights

            breakpoint()
            # Feature map before global pooling: BNC -> BCHW
            feature_map = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            
        return feature_map

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/bigdata/SurgPose/tooltip/l_mask/frame072.png")).unsqueeze(0).to(device)
text = clip.tokenize(["left gripper tips of the surgical instrument", "right gripper tips of the surgical instrument"]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    print(text_features.shape)

# get attention maps
# attn_maps = get_attention_maps(model, image)
# print("Attention map shape:", attn_maps.shape)

# breakpoint()

feature_extractor = FeatureExtractor(model).to(device)

feature_map = feature_extractor(image)  # [1, C, H, W]
print("Feature map shape:", feature_map.shape)  # [1, C, H, W]


# image_features = model.encode_image(image)

# print(image_features.shape, text_features.shape)

# logits_per_image, logits_per_text = model(image, text)
# probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]