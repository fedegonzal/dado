from transformers import  DPTForDepthEstimation, DPTImageProcessor, DPTFeatureExtractor
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_model():
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-base-384")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-beit-base-384")
    return model, feature_extractor

# Step 2: Obtain depth map
def get_depth_map(image, model, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return predicted_depth


# Uses Huggingface transformers to obtain 
# Depth prediction from a image
def get_depth_prediction(pil_img, model):

    image_processor = DPTImageProcessor.from_pretrained(model)
    depth_model = DPTForDepthEstimation.from_pretrained(model)

    inputs = image_processor(images=pil_img, return_tensors="pt")

    # forward pass
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    return prediction, predicted_depth


# Function to create overlapping depth bins
def create_overlapping_depth_bins(depth_map, num_bins=5, overlap=0.2):
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    bin_edges = np.linspace(depth_min, depth_max, num_bins + 1)

    # Create overlapping bins
    adjusted_bins = []
    for i in range(num_bins):
        start = max(depth_min, bin_edges[i] - overlap * (bin_edges[i + 1] - bin_edges[i]))
        end = min(depth_max, bin_edges[i + 1] + overlap * (bin_edges[i + 1] - bin_edges[i]))
        adjusted_bins.append((start, end))

    return adjusted_bins

# Function to segment the depth map into bins
def segment_depth_map(depth_map, bins):
    segmented_map = np.zeros_like(depth_map)
    for i, (start, end) in enumerate(bins):
        mask = (depth_map >= start) & (depth_map < end)
        segmented_map[mask] = i + 1  # Assign a unique label for each bin
    return segmented_map


# Function to visualize results
def visualize_results(original_image, depth_map, segmented_map):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show original image
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    # Show depth map
    axs[1].imshow(depth_map, cmap='inferno')
    axs[1].set_title("Depth Map")
    axs[1].axis("off")

    # Show segmented depth map
    axs[2].imshow(segmented_map, cmap='jet')
    axs[2].set_title("Segmented Depth Bins")
    axs[2].axis("off")

    plt.show()
