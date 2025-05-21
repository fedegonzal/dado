import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import matplotlib.colors as mcolors

from myutils.depth import get_depth_map, load_model
from myutils.utils import load_image

def compute_entropy_patches(image, n=4, alpha=0.5, add_text=True):
    width, height = image.size
    patch_w, patch_h = width // n, height // n

    # Create overlay image for drawing transparent patches
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Store entropy values
    entropy_values = []

    # Compute entropy per patch
    for i in range(n):
        for j in range(n):
            left = j * patch_w
            top = i * patch_h
            right = (j + 1) * patch_w
            bottom = (i + 1) * patch_h
            patch = np.array(image.crop((left, top, right, bottom)))
            entropy = shannon_entropy(patch)
            entropy_values.append((i, j, entropy))

    # Normalize entropy values between 0 and 1
    entropies = np.array([e[2] for e in entropy_values])
    norm_entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)

    # Draw overlay with colors and text
    for idx, (i, j, entropy) in enumerate(entropy_values):
        norm_entropy = norm_entropies[idx]
        # Interpolate color between blue (low) and red (high)
        alpha = 1 - norm_entropy
        color = tuple(int(c * 255) for c in mcolors.to_rgb(plt.cm.bwr(norm_entropy))) + (int(alpha * 255),)
        left = j * patch_w
        top = i * patch_h
        right = (j + 1) * patch_w
        bottom = (i + 1) * patch_h
        draw.rectangle([left, top, right, bottom], fill=color)

        # Add text in the middle of the patch
        if add_text:
            text = f"{norm_entropy:.2f}"
            font_size = min(patch_w, patch_h) // 5
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            text_x = left + (patch_w - text_w) / 2
            text_y = top + (patch_h - text_h) / 2
            draw.text((text_x, text_y), text, fill="black", font=font)

    # Combine original image and overlay
    result_image = Image.alpha_composite(image.convert("RGBA"), overlay)

    # Show image
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.axis("off")
    plt.title(f"Patch Entropy Map (n={n})")
    plt.show()

# Example usage:
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000030.jpg"
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000340.jpg"
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/002215.jpg"
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000372.jpg"
image_path = "datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/000396.jpg"

# Load image
image = load_image(image_path)

compute_entropy_patches(image, n=32, alpha=0.5, add_text=False)

exit()

# depth is (n,m) convert to PIL image
model, feature_extractor = load_model()
depth = get_depth_map(image, model, feature_extractor)
depth = Image.fromarray(depth)

compute_entropy_patches(depth, n=16, alpha=0.5)

# dino

from myutils.dino2 import load_dino2_model
