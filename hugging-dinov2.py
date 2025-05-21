import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# === Step 1: Load image and model ===
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000030.jpg"
image = Image.open(image_path).convert("RGB")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base", output_attentions=True)
model.eval()

# === Step 2: Preprocess the image ===
inputs = processor(images=image, return_tensors="pt")

# === Step 3: Forward pass with attentions ===
with torch.no_grad():
    outputs = model(**inputs)

# === Step 4: Extract CLS attention ===
# outputs.attentions is a tuple of [layer_count x (batch, heads, tokens, tokens)]
# We use the last layer
attn = outputs.attentions[-1]  # shape: (1, heads, tokens, tokens)
cls_attn = attn[0, :, 0, 1:]   # CLS token to all patches (heads, patches)
cls_attn = cls_attn.mean(0).cpu().numpy()  # Mean over heads

# === Step 5: Reshape attention map to 2D ===
# Number of tokens = number of patches (e.g., 14x14 for base)
num_patches = cls_attn.shape[0]
feat_size = int(num_patches ** 0.5)
attn_map = cls_attn.reshape(feat_size, feat_size)

# === Step 6: Normalize and resize to original image ===
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
attn_map_resized = cv2.resize(attn_map, image.size, interpolation=cv2.INTER_CUBIC)

# === Step 7: Overlay attention on image ===
plt.figure(figsize=(8, 8))
#plt.imshow(image)
plt.imshow(attn_map_resized)
plt.axis('off')
plt.title("DINOv2 CLS Attention")
plt.show()
