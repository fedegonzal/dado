import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

# 1. Load the pre-trained DINOv2 model
# We'll use a smaller model for demonstration purposes (vits14)
# You can change to vitb14, vitl14, vitg14 depending on your VRAM and desired performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

# DINOv2 models are typically trained on images scaled to 518x518 or 448x448
# and use ImageNet normalization. Patch size for vits14 is 14.
# The output feature map resolution will be original_size / patch_size.
# We'll resize the input image.
transform = T.Compose([
    T.Resize((518, 518)), # Resize input image
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --- Configuration ---
# Replace with your image path
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000396.jpg"

n_components_pca = 50  # Number of PCA components to keep
n_clusters_kmeans = 10  # Number of clusters for K-Means (discovered "objects"/regions)
patch_size = 14 # Patch size for dinov2_vits14

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
    # Get patch features. The model output is a dictionary.
    # 'x_norm_patchtokens' contains the normalized patch features.
    features = model.forward_features(img_tensor)
    patch_features = features['x_norm_patchtokens'] # Shape: [1, num_patches, feature_dim]

# Reshape features: Remove batch dim, then flatten spatial dims
# [1, H*W, feature_dim] -> [H*W, feature_dim]
num_patches = patch_features.shape[1]
feature_dim = patch_features.shape[2]
patch_features_flat = patch_features.squeeze(0).cpu().numpy() # Move to CPU for numpy/sklearn
print(f"Extracted {num_patches} patch features with dimension {feature_dim}.")

# Calculate spatial dimensions of the feature map
img_height, img_width = img_tensor.shape[-2:] # Get dimensions of processed image
feature_h = img_height // patch_size
feature_w = img_width // patch_size
print(f"Feature map spatial resolution: {feature_h}x{feature_w}")


# --- Apply PCA ---
print(f"Applying PCA to reduce feature dimension to {n_components_pca}...")
pca = PCA(n_components=n_components_pca)
patch_features_pca = pca.fit_transform(patch_features_flat)
print(f"Reduced features shape: {patch_features_pca.shape}")


# --- Apply K-Means Clustering ---
print(f"Applying K-Means clustering with {n_clusters_kmeans} clusters...")
kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0, n_init=10) # n_init to avoid local minima
cluster_labels = kmeans.fit_predict(patch_features_pca)
print("Clustering complete.")

# Reshape cluster labels back to the spatial grid
cluster_grid = cluster_labels.reshape((feature_h, feature_w))


# --- Create Mask Image ---
print("Creating mask image...")

# Generate random colors for each cluster (excluding black for background)
colors = [[0, 0, 0]] # Black for cluster 0 (often the largest, might be background)
for _ in range(n_clusters_kmeans - 1):
    # Generate distinct random colors
    colors.append([random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)])

# Create an empty mask image with the feature map spatial resolution
mask_img_small = np.zeros((feature_h, feature_w, 3), dtype=np.uint8)

# Assign colors based on cluster labels
for i in range(feature_h):
    for j in range(feature_w):
        mask_img_small[i, j] = colors[cluster_grid[i, j]]

# Upsample the mask to the original input image size for better visualization
# Using nearest neighbor interpolation to keep hard boundaries between segments
mask_img_full_size = Image.fromarray(mask_img_small, 'RGB').resize(img.size, Image.Resampling.NEAREST)
mask_img_full_size_np = np.array(mask_img_full_size)

print("Mask image created.")

# --- Display Results ---
print("Displaying results.")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(mask_img_full_size_np)
axes[1].set_title(f"Object Isolation (PCA={n_components_pca}, KMeans k={n_clusters_kmeans})")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# You can also save the mask image
# mask_img_full_size.save('object_mask.png')
# print("Mask image saved as object_mask.png")
