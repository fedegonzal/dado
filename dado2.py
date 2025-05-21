from dado2.depth import DepthEstimator
from PIL import Image
import matplotlib.pyplot as plt

image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000396.jpg"

pil_image = Image.open(image_path).convert("RGB")

# https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads
# Models: Intel/dpt-beit-base-384, Intel/dpt-large, facebook/dpt-dinov2-base-kitti
depth_estimator = DepthEstimator(model_name="Intel/dpt-large", device="mps")

depth_map = depth_estimator.predict_depth(pil_image)

plt.imshow(depth_map, cmap="viridis") # other options: "magma", "plasma", "inferno", "cividis"
plt.axis("off")
plt.show()

depth_rgb = depth_estimator.to_rgb_image(colormap="viridis")

plt.imshow(depth_rgb)
plt.axis("off")
plt.show()

# depth_estimator.save("depth_map3.png")

# ----------------------

# Load model directly
from transformers import AutoImageProcessor, AutoModel
import torch

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = AutoModel.from_pretrained("facebook/dinov2-large", output_attentions=True)

model.eval()

inputs = processor(images=pil_image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get attention from last layer
# Shape: (batch, num_heads, num_tokens, num_tokens)
attn = outputs.attentions[-1]
attn_cls = attn[0, :, 0, 1:]  # CLS token attending to all other tokens

# Average over heads
attn_map = attn_cls.mean(0).reshape(16, 16)

import torch.nn.functional as F
import matplotlib.pyplot as plt

# Resize heatmap to input image size
heatmap = attn_map.unsqueeze(0).unsqueeze(0)  # shape (1, 1, 14, 14)
heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
heatmap = heatmap.squeeze().numpy()

# Display with overlay
#plt.imshow(pil_image.resize((224, 224)))
plt.imshow(heatmap)
plt.axis('off')
plt.title("CLS Attention Heatmap (Hugging Face DINOv2)")
plt.show()


inputs = processor(images=pil_image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

# Get the CLS token
cls_token = last_hidden_states[:, 0, :]  # shape (batch_size, hidden_size)
cls_token = cls_token.squeeze().numpy()  # shape (hidden_size,)
# Normalize the CLS token
cls_token = (cls_token - cls_token.min()) / (cls_token.max() - cls_token.min() + 1e-8)
# Convert to uint8
cls_token_uint8 = (cls_token * 255).astype(np.uint8)
# Convert to PIL Image
cls_token_image = Image.fromarray(cls_token_uint8, mode='L')
plt.imshow(cls_token_image)
plt.axis('off')
plt.title("CLS Token (Hugging Face DINOv2)")
plt.show()
