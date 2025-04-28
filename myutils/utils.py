import os
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Function to count the number of unique colors in an image
def count_colors(image):
    # convert to 1D array
    reshaped = image.flatten()
    unique = np.unique(reshaped, axis=0)
    print(unique)
    return len(unique)


import numpy as np

# Define IoU calculation
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate the area of intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Apply Non-Maximum Suppression (NMS)
def non_maximum_suppression(boxes, iou_threshold=0.5):
    boxes = sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)  # Sort by area
    filtered_boxes = []
    
    while boxes:
        # Select the box with the largest area (or confidence if available)
        current_box = boxes.pop(0)
        filtered_boxes.append(current_box)
        
        # Filter out boxes that have a high IoU with the current box
        boxes = [box for box in boxes if calculate_iou(current_box, box) < iou_threshold]
    
    return filtered_boxes


def filter_bounding_boxes(boxes):
    # Helper function to check if box1 is inside box2
    def is_inside(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return (x1 >= x2 and y1 >= y2 and 
                (x1 + w1) <= (x2 + w2) and 
                (y1 + h1) <= (y2 + h2))
    
    # Result list for boxes that are not inside any other box
    result = []
    for i, box1 in enumerate(boxes):
        inside_any = False
        for j, box2 in enumerate(boxes):
            if i != j and is_inside(box1, box2):
                inside_any = True
                break
        if not inside_any:
            result.append(box1)
    return result


# Function to calculate histogram peaks and dynamically create bins based on peaks
def create_dynamic_depth_bins(depth_map, overlap=0.2, peak_prominence=0.05, bin_width=0.1, bins=100):
    # Calculate histogram of depth values
    depth_hist, bin_edges = np.histogram(depth_map, bins=bins)

    # Identify peaks in the histogram (potential depth layers)
    peaks, _ = find_peaks(depth_hist, prominence=peak_prominence * np.max(depth_hist))

    #print(f"Found {len(peaks)} peaks in the histogram.")
    
    # Calculate dynamic bins based on the peak locations
    depth_min, depth_max = np.min(depth_map), np.max(depth_map)
    adjusted_bins = []
    for peak in peaks:
        center = bin_edges[peak]
        width = bin_width * (depth_max - depth_min)
        start = max(depth_min, center - width * (1 + overlap))
        end = min(depth_max, center + width * (1 + overlap))
        adjusted_bins.append((start, end))
        
    return adjusted_bins, depth_hist


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def load_image_as_cv2(image_path):
    """ Function printing python version. """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_as_pil(image_path):
    """ Function printing python version. """
    img = Image.open(image_path)
    return img


def load_image_as_tensor(pil_img, size=600):
    """ Function printing python version. """
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    return transform(pil_img)


def img_tensor_padded(img_tensor, patch_size):
    """ Function printing python version. """
    new_image_container = (
        img_tensor.shape[0],
        int(np.ceil(img_tensor.shape[1] / patch_size) * patch_size),
        int(np.ceil(img_tensor.shape[2] / patch_size) * patch_size),
    )
    img_paded = torch.zeros(new_image_container)
    img_paded[:, : img_tensor.shape[1], : img_tensor.shape[2]] = img_tensor

    return img_paded



def list_filenames_without_extension(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filename_without_extension = os.path.splitext(filename)[0]
            filenames.append(filename_without_extension)
    return filenames



def plot_image_grid(title, pil_img, depth_image_resized, sum_atts_resized, final_attention_map, final_att_thresholded, final_image):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    # set a title for the figure
    fig.suptitle(title, fontsize=16)

    axs[0, 0].imshow(pil_img)
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(depth_image_resized)
    axs[0, 1].set_title('Depth image')

    axs[0, 2].imshow(sum_atts_resized)
    axs[0, 2].set_title(f'Dino Attention')

    axs[1, 0].imshow(final_attention_map)
    axs[1, 0].set_title('Final Attention Map')

    axs[1, 1].imshow(final_att_thresholded)
    axs[1, 1].set_title('Final Attention Map Thresholded')

    axs[1, 2].imshow(final_image)
    axs[1, 2].set_title('Final Image')

    plt.show()
