import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

# === Step 1: Load image and model ===
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000030.jpg"
image = Image.open(image_path).convert("RGB")

# === Step 2: Load DPT-hybrid model and processor ===
processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid")
model.eval()

# === Step 3: Preprocess ===
inputs = processor(images=image, return_tensors="pt")

# === Step 4: Predict depth ===
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # [1, 1, H, W]

# === Step 5: Post-processing ===
# Convert to numpy and squeeze
depth = predicted_depth.squeeze().cpu().numpy()

# Normalize to [0, 1]
depth_min, depth_max = depth.min(), depth.max()
depth_normalized = (depth - depth_min) / (depth_max - depth_min)

# Resize to original image size
depth_resized = cv2.resize(depth_normalized, image.size, interpolation=cv2.INTER_CUBIC)

# --- Optional Post-processing ---

# (A) Gaussian Blur to smooth transitions
depth_blurred = cv2.GaussianBlur(depth_resized, (11, 11), sigmaX=0)

# (B) Histogram Equalization to normalize contrast
depth_eq = cv2.equalizeHist((depth_blurred * 255).astype(np.uint8)) / 255.0

# === Step 6: Display ===
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(depth_resized, cmap='inferno')
plt.title("Raw Depth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(depth_eq, cmap='inferno')
plt.title("Post-Processed Depth")
plt.axis("off")

plt.tight_layout()
plt.show()
