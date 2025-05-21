import torch
import numpy as np
import cv2
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from skimage.segmentation import slic
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# === 1. Load RGB Image ===
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000396.jpg"

rgb_path = image_path
rgb_cv = cv2.imread(rgb_path)
rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)  # OpenCV loads in BGR
rgb_pil = Image.fromarray(rgb_cv)

# === 2. Use DPT to Estimate Depth ===
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Prepare input
inputs = feature_extractor(images=rgb_pil, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Resize to original size
depth = predicted_depth.squeeze().cpu().numpy()
depth_resized = cv2.resize(depth, (rgb_cv.shape[1], rgb_cv.shape[0]))
depth_norm = cv2.normalize(depth_resized, None, 0, 1, cv2.NORM_MINMAX)

# Assume depth_normalized is (H, W), float32, in [0, 1]
depth_3ch = np.stack([depth_norm]*3, axis=-1)  # Shape: (H, W, 3)

from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

# Let´s apply watershed segmentation to the depth map
# https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background

depth_binary = depth_norm > 0.1  # Thresholding to create a binary image
depth_binary = depth_binary.astype(np.uint8)  # Convert to uint8
depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # Close small holes
depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # Remove small objects
depth_binary = cv2.dilate(depth_binary, np.ones((3, 3), np.uint8), iterations=1)  # Dilate to fill gaps
depth_binary = cv2.erode(depth_binary, np.ones((3, 3), np.uint8), iterations=1)  # Erode to remove noise
depth_binary = cv2.GaussianBlur(depth_binary, (5, 5), 0)  # Gaussian blur to smooth edges
depth_binary = cv2.threshold(depth_binary, 0.5, 1, cv2.THRESH_BINARY)[1]  # Thresholding to create a binary image
depth_binary = depth_binary.astype(np.uint8)  # Convert to uint8

distance = ndi.distance_transform_edt(depth_binary)
coords = peak_local_max(distance)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=depth_binary)

fig, axes = plt.subplots(ncols=3, figsize=(12, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(depth_binary)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance)
ax[1].set_title('Distances')
ax[2].imshow(labels)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()

# exit()

# let´s apply Watershed segmentation to the depth_3ch

# === 2.1 Apply Watershed Segmentation ===
# Convert to grayscale
depth_gray = cv2.cvtColor(depth_3ch, cv2.COLOR_RGB2GRAY)
# Thresholding to create a binary image
_, binary = cv2.threshold(depth_gray, 0.5, 255, cv2.THRESH_BINARY_INV)
# Distance transform
dist_transform = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
# Normalize the distance transform
dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
# Thresholding the distance transform
_, sure_fg = cv2.threshold(dist_transform, 0.7, 255, 0)
# Finding unknown region
unknown = cv2.subtract(binary.astype(np.uint8), sure_fg.astype(np.uint8))
# Marker labelling
_, markers = cv2.connectedComponents(binary.astype(np.uint8))
markers = markers + 1  # Increment all labels by 1
markers[unknown == 255] = 0  # Mark the unknown region with zero
# Apply watershed
markers = cv2.watershed(depth_3ch.astype(np.uint8), markers)
# Mark the boundaries
depth_3ch[markers == -1] = [255, 0, 0]  # Mark boundaries in red
# Display the result
plt.imshow(depth_3ch)
plt.axis("off")
plt.title("Watershed Segmentation on Depth")
plt.show()





# apply superpixel segmentation to the depth map and display it

# Assume depth_normalized is (H, W), float32, in [0, 1]
depth_3ch = np.stack([depth_norm]*3, axis=-1)  # Shape: (H, W, 3)

# Apply SLIC
superpixels = slic(depth_3ch, n_segments=100, compactness=0.1, start_label=1)


from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

boundaries = mark_boundaries(depth_norm, superpixels)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(depth_norm)
plt.axis("off")
plt.title("Estimated Depth")

plt.subplot(1, 3, 2)
plt.imshow(superpixels)
plt.axis("off")
plt.title("Superpixels on Depth")

plt.subplot(1, 3, 3)
plt.imshow(boundaries)
plt.axis("off")
plt.title("Superpixels Boundaries")

plt.show()

exit()

# === 3. Apply SLIC Superpixels on RGB + Depth ===
img_4d = np.dstack((rgb_cv, depth_norm))  # shape: (H, W, 4)
superpixels = slic(img_4d, n_segments=300, compactness=10, start_label=1)


# === 4. Feature Extraction per Superpixel ===
features = []
labels = np.unique(superpixels)
for label in labels:
    mask = (superpixels == label)
    mean_color = rgb_cv[mask].mean(axis=0)
    mean_depth = depth_norm[mask].mean()
    yx = np.argwhere(mask).mean(axis=0)
    features.append(np.hstack([mean_color, mean_depth, yx[::-1]]))  # [R, G, B, D, x, y]

features = np.array(features)


# === 5. Clustering with DBSCAN ===
clustering = DBSCAN(eps=10, min_samples=5).fit(features)
cluster_labels = clustering.labels_


# === 6. Create Mask from Clusters ===
output_mask = np.zeros_like(superpixels)
for sp_label, cluster_id in zip(labels, cluster_labels):
    if cluster_id == -1:
        continue  # ignore noise
    output_mask[superpixels == sp_label] = cluster_id + 1

# display the mask
plt.imshow(output_mask)
plt.axis("off")
plt.title("Unsupervised Object Masks")
plt.show()


# === 7. Colorize and Visualize ===
def colorize_mask(mask):
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    n_labels = mask.max()
    colors = plt.cm.get_cmap('tab20', n_labels + 1)
    for i in range(1, n_labels + 1):
        output[mask == i] = (np.array(colors(i)[:3]) * 255).astype(np.uint8)
    return output

mask_rgb = colorize_mask(output_mask)
blended = cv2.addWeighted(rgb_cv, 0.6, mask_rgb, 0.4, 0)

# === 8. Show Results ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("RGB")
plt.imshow(rgb_cv)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Estimated Depth")
plt.imshow(depth_norm, cmap="inferno")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Unsupervised Object Masks")
plt.imshow(blended)
plt.axis("off")

plt.tight_layout()
plt.show()
