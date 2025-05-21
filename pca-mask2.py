import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt
import random

# 1. Load the pre-trained DINOv2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

transform = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --- Configuration ---
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000396.jpg"

n_components_pca = 50
patch_size = 14

# --- Load and Preprocess Image ---
try:
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(f"Loaded image from {image_path} and preprocessed.")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()
except Exception as e:
    print(f"Error loading or processing image: {e}")
    exit()

# --- Extract DINOv2 Features ---
print("Extracting DINOv2 features...")
with torch.no_grad():
    features = model.forward_features(img_tensor)
    patch_features = features['x_norm_patchtokens']

num_patches = patch_features.shape[1]
feature_dim = patch_features.shape[2]
patch_features_flat = patch_features.squeeze(0).cpu().numpy()
print(f"Extracted {num_patches} patch features with dimension {feature_dim}.")

img_height, img_width = img_tensor.shape[-2:]
feature_h = img_height // patch_size
feature_w = img_width // patch_size
print(f"Feature map spatial resolution: {feature_h}x{feature_w}")

# --- Apply PCA ---
print(f"Applying PCA to reduce feature dimension to {n_components_pca}...")
pca = PCA(n_components=n_components_pca)
patch_features_pca = pca.fit_transform(patch_features_flat)
print(f"Reduced features shape: {patch_features_pca.shape}")

# --- Apply HDBSCAN Clustering ---
print("Applying HDBSCAN clustering (automatic number of clusters)...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels = clusterer.fit_predict(patch_features_pca)
n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Clustering complete. Found {n_clusters_found} clusters (excluding noise).")

# --- Create Mask Image ---
print("Creating mask image...")

unique_labels = set(cluster_labels)
colors = {}
for label in unique_labels:
    if label == -1:
        colors[label] = [0, 0, 0]  # Black for noise
    else:
        colors[label] = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]

mask_img_small = np.zeros((feature_h, feature_w, 3), dtype=np.uint8)
for idx, label in enumerate(cluster_labels):
    i = idx // feature_w
    j = idx % feature_w
    mask_img_small[i, j] = colors[label]

mask_img_full_size = Image.fromarray(mask_img_small, 'RGB').resize(img.size, Image.Resampling.NEAREST)
mask_img_full_size_np = np.array(mask_img_full_size)

# --- Display Results ---
print("Displaying results.")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(mask_img_full_size_np)
axes[1].set_title(f"Object Isolation (PCA={n_components_pca}, HDBSCAN clusters={n_clusters_found})")
axes[1].axis('off')

plt.tight_layout()
plt.show()
