import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_normals(depth):
    # Calculate gradients
    dzdx = np.gradient(depth, axis=1)
    dzdy = np.gradient(depth, axis=0)

    # Create the normal vectors
    normals = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    # Normalize the normals
    norms = np.linalg.norm(normals, axis=2)
    normals /= norms[..., np.newaxis]
    return normals

def normalize_to_color(normals):
    # Map normals to color values
    normals_color = (normals + 1) / 2 * 255  # Normalize to [0, 255]
    normals_color = normals_color.astype(np.uint8)
    return normals_color


# Load the model and processor
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")
processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-base-384")

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load an image
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000017.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt").to(device)

# Get depth predictions
with torch.no_grad():
    outputs = model(**inputs)

# Get the depth map (it's returned as logits)
depth_map = outputs.predicted_depth[0].cpu().numpy()

normals = compute_normals(depth_map)

normals_colored = normalize_to_color(normals)

# Display the depth map and the normal map
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(depth_map, cmap='gray')
plt.title("Depth Map")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(normals_colored)
plt.title("Normal Map")
plt.axis('off')

plt.show()
