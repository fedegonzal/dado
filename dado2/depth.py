import torch
import numpy as np
from PIL import Image
from transformers import DPTForDepthEstimation, AutoImageProcessor
import matplotlib.cm as cm


class DepthEstimator:

    def __init__(self, model_name: str = "Intel/dpt-large", device: str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.original_size = (None, None)  # Placeholder for original size
        self.depth_map = None  # Placeholder for depth map
        self.colored_depth_uint8 = None # Placeholder for colored depth map

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict_depth(self, pil_image):

        # Store the image original size
        self.original_size = pil_image.size

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        self.depth_map = prediction.squeeze().cpu().numpy()

        return self.depth_map
    

    def to_rgb_image(self, colormap: str = "viridis") -> Image:
        # Normalize depth to [0, 1]
        depth_min = np.min(self.depth_map)
        depth_max = np.max(self.depth_map)
        norm_depth = (self.depth_map - depth_min) / (depth_max - depth_min + 1e-8)

        # Apply the viridis colormap (RGBA output)
        colormap = cm.get_cmap(colormap)
        colored_depth = colormap(norm_depth)[:, :, :3]  # Drop alpha channel

        # Convert to uint8 and then to PIL Image
        self.colored_depth_uint8 = (colored_depth * 255).astype(np.uint8)
        return Image.fromarray(self.colored_depth_uint8)



    def save(self, path: str):
        if self.colored_depth_uint8 is None:
            self.to_rgb_image()

        # Save the colored depth map
        Image.fromarray(self.colored_depth_uint8).save(path)
        print(f"Depth map saved to {path}")
