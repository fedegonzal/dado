import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

# Cargar imagen de ejemplo
dataset_path = "datasets/VOC2007/VOCdevkit/VOC2007"
image = "009331.jpg"
image_path = f"{dataset_path}/JPEGImages/{image}"
image = Image.open(image_path).convert("RGB")

# Preprocesamiento como espera DINO
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

img_tensor = transform(image).unsqueeze(0)  # Agrega batch dimension

# Cargar modelo DINO
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

# Extraer las features
with torch.no_grad():
    features = model(img_tensor)  # features.shape -> (1, 384) o similar

import matplotlib.pyplot as plt
import seaborn as sns

# features: tensor de forma (1, 384)
features_np = features.squeeze().numpy()

plt.figure(figsize=(10, 1))
sns.heatmap([features_np], cmap='viridis', cbar=True)
plt.title("DINO feature vector (384D)")
plt.xlabel("Feature index")
plt.yticks([])
plt.show()
