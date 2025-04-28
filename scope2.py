# 1. read an image
# 2. convert to tensor and transform
# 3. padding to prepare for dino model

# 6. pass through the model to get the depth
# 7. obtain the depth map

# 4. pass through dino model to get features
# 5. obtain the attention map

# 8. process depth (get the attention)
# 9. obtain the attention map from depth processed
# 10. process the attention map
# 11. obtain the final attention map merged from depth and dino

import torch
from torch import nn
from PIL import Image

from tqdm import tqdm

from myutils.depth import *
from myutils.dino1 import load_dino1_model
from myutils.dino2 import load_dino2_model
from myutils.discovery import *
from myutils.pascal_voc import *
from myutils.ssl import *
from myutils.utils import *
from myutils.datasets import bbox_iou

import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

import base64
import json
import argparse
from datetime import datetime

from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import numpy as np
from scipy.signal import find_peaks


# Read the arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, help='prefix results file to store the results')
    args = parser.parse_args()
    return args


# Prepares the results folder and returns the csv filename
def setup_results_folder(timestamp):
    os.makedirs(f"results/{timestamp}_images", exist_ok=True)
    results_filename = f"results/{timestamp}_results.csv"
    return results_filename

# Open the results file in read/write mode
def load_results_file(results_filename: str):
    mode = 'r+' if os.path.exists(results_filename) else 'w+'
    return open(results_filename, mode)

# Get the images from the dataset
# as a list of image names: ['000001', '000002', ...]
def get_images_from_dataset(dataset):
    # Get N random images from dataset_path
    # n=None to get all images

    #images = ['005881', '000250', '001855', '007586', '009079', '003159', '009745', '004801', '004244', '006185', '002405', '000334', '008349', '007853', '001263', '007193', '008856', '003678', '008833', '000109', '002873', '009881', '007040', '007751', '008917', '003369', '003629', '004828', '004192', '008415', '008733', '009418', '006694', '005676', '006339', '001590', '004281', '008096', '009438', '006212', '005064', '008372', '009343', '001185', '009307', '003433', '002266', '009214', '009421', '007417', '000951', '004359', '005732', '007697', '001981', '003147', '002224', '003051', '004706', '000047', '001393', '001510', '002977', '003994', '002880', '005790', '006247', '006282', '004441', '006822', '006626', '007398', '007261', '009246', '007503', '001343', '001250', '003807', '000667', '003120', '000489', '007372', '001455', '000470', '004797', '005101', '000917', '005312', '007847', '009904', '006425', '000483', '001927', '009331', '004518', '000173', '002439', '005365', '007621', '006968']
    #images = ['009331', '002439', '005365', '007621', '006968']
    #images = ['009331']
    #images = ['001258']

    if dataset == 'VOC2007':
        dataset_path = "datasets/VOC2007/VOCdevkit/VOC2007"
    
    elif dataset == 'VOC2012':
        dataset_path = "datasets/VOC2012/VOCdevkit/VOC2012"

    elif dataset == 'COCO':
        dataset_path = "datasets/COCO/val2017"
    
    else:
        raise ValueError("Invalid dataset")
    
    return get_images(dataset_path, -1), dataset_path


# Get the image and annotation path
def get_image_anno_path(img_name, dataset_path):
    image_path = f'{dataset_path}/JPEGImages/{img_name}.jpg'
    annotation_path = f'{dataset_path}/Annotations/{img_name}.xml'
    return image_path, annotation_path


args = parse_args()

# dino v1
params = {
    'patch_size': 16,
    'ssl_checkpoint': 'pretrained/dino_deitsmall16_pretrain.pth',
#    'depth_checkpoint': 'Intel/dpt-hybrid-midas',
    'depth_checkpoint': 'Intel/dpt-beit-base-384',
    'img_size': None
}

# dino v2
#params = {
#    'patch_size': 14,
#    'ssl_checkpoint': 'pretrained/dinov2_vits14_reg4_pretrain.pth',
#    'depth_checkpoint': 'Intel/dpt-beit-base-384',
#    'img_size': 526
#}

# Set the dataset
dataset = 'VOC2007'

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (255, 255, 0),  # Yellow
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 20, 147), # Pink
    (0, 128, 0),    # Dark Green
    (139, 69, 19)   # Brown
]

# let's create or open a file to store the results, we want to read and write

# timestamp has the format: YYYYMMDD, else today
timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d")

results_filename = setup_results_folder(timestamp)

results_file = load_results_file(results_filename)

images, dataset_path = get_images_from_dataset(dataset)

corloc = np.zeros(len(images))

print("Processing images...")

# initialize tqdm with a progress bar
progress_bar = tqdm(images)

# we will use this flag to stop reading the file
# in case we reach the end of the file
continue_reading = True


# let's use tqdm to show a progress bar
for i, img_name in enumerate(progress_bar):

    if (continue_reading):

        # read the line i from the results_file
        line = results_file.readline()

        if (line):

            img_name_file = line.split(",")[1]

            # if img_name is in results_file, we skip it
            if (img_name in line):

                # get the corloc from the line
                corloc[i] = float(line.split(",")[2])
                local_corloc = corloc[i]

                # corloc_now is the mean of the corloc until now
                corloc_now = corloc[:i+1].sum() / (i+1)

                # show the image name in the progress bar without breaking the tqdm
                progress_bar.set_description(f"Processed {img_name} // Image corloc: {local_corloc:.2f} // Corloc now: {corloc_now:.2f}")

                # skip to the next image
                continue
        
        # if we reach the end of the file, 
        # we stop reading
        else:
            continue_reading = False


    # The image isn't in the previous results, 
    # let's process it

    image_path, annotation_path = get_image_anno_path(img_name, dataset_path)

    ##############
    # WARMING UP #
    ##############

    # Read an image
    pil_img = load_image_as_pil(image_path)

    # Convert to tensor and transform
    img_tensor = load_image_as_tensor(pil_img).to(device)

    # Get the ground truth
    ground_truth, ground_truth_img = get_ground_truth_voc2007(annotation_path, pil_img)

    #plt.imshow(ground_truth_img)
    #plt.show()

    #########################
    # GETTING THE ATTENTION #
    #########################

    # Padding the image with zeros to fit multiple of patch-size
    patch_size = params['patch_size']
    img_paded = img_tensor_padded(img_tensor, patch_size).to(device)

    # Load the SSL model
    model = load_dino1_model(patch_size, params['ssl_checkpoint'], device, params['img_size'])
    #model = load_dino2_model(patch_size, params['ssl_checkpoint'], device, params['img_size'])
    model.to(device)
    model.eval()

    # Get the attentions
    atts = get_attentions(model, img_paded, patch_size)

    # Obtain the attention map

    # Sum the attention outputs (6 outputs used in DINO)
    #sum_atts = atts.sum(0)
    sum_atts = atts.max(0)

    # resize the attention map to the original image size
    # resize the image to the original size
    sum_atts = cv2.resize(sum_atts, (pil_img.size[0], pil_img.size[1]))

    # Normalize the attention map to 0-1
    atts_norm = sum_atts / np.max(sum_atts)


    #plt.imshow(atts_norm)
    #plt.show()




    #################
    # GETTING DEPTH #
    #################

    image = load_image(image_path)

    #plt.imshow(image)
    #plt.show()

    #print("Image shape:", image.size)

    model, feature_extractor = load_model()

    # Generate depth map
    depth_map = get_depth_map(image, model, feature_extractor)

    # resize the image to the original size
    depth_map = cv2.resize(depth_map, (image.size[0], image.size[1]))

    #plt.imshow(depth_map)
    #plt.show()



    # Get image dimensions
    height, width = depth_map.shape

    # Define the vertical region for modification (75% to 100% of height)
    start_row = int(height * 0.75)

    # Create a scaling mask for the lower 25% of the image
    scaling_factor = np.linspace(1, 0.5, height - start_row)  # Scale down from 1 to 0.5
    scaling_mask = np.ones((height, width))

    # Apply the scaling mask to the target region
    for idx, factor in enumerate(scaling_factor):
        scaling_mask[start_row + idx, :] = factor

    # let's normalize the depth map to 0-1
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    #print("depth_map:", depth_map.min(), depth_map.max())
    #print("scaling_mask:", scaling_mask.min(), scaling_mask.max())

    # Apply the mask to scale down the depth values in the target area
    modified_depth_map = depth_map * scaling_mask

    # Clip values to maintain within valid range for depth data type
    modified_depth_map = np.clip(modified_depth_map, 0, np.max(depth_map)).astype(depth_map.dtype)

    #plt.imshow(modified_depth_map)
    #plt.show()

    depth_map = modified_depth_map.copy()

    # normalize the depth map to 0-255
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)


    # Assuming depth_map is your depth data array
    bins, hist = create_dynamic_depth_bins(depth_map, overlap=0.2, peak_prominence=0.05, bin_width=0.1)

    #print("Bins:", bins)

    #plt.hist(depth_map.flatten(), bins=100)
    #plt.show()


    # Calculate histogram with a sufficient range to cover all values
    hist_range = (0, depth_map.max())  # Adjust the range based on max pixel value
    hist, bin_edges = np.histogram(depth_map, bins=12, range=hist_range)

    #print("Histogram:", hist)
    #print("Bin Edges:", bin_edges)

    segmented_map = depth_map.copy()
    # let's assign 0 to values below 50
    #segmented_map[segmented_map < 30] = 0

    #segmented_map = np.zeros_like(depth_map)
    #for i, bin in enumerate(bins):
    #    mask = (depth_map >= bin)
    #    segmented_map[mask] = i + 1  # Assign a unique label for each bin
        
    #plt.imshow(segmented_map)
    #plt.show()

    # let's convert segmented_map to 9 gray levels
    segmented_map = cv2.normalize(segmented_map, None, 0, 9, cv2.NORM_MINMAX)
    segmented_map = segmented_map.astype(np.uint8)

    #plt.imshow(segmented_map)
    #plt.show()



    #print("unique values segmented_map:", np.unique(segmented_map))

    simplified = segmented_map.copy()

    # pixels with values 0, 1, 2, 3 are set to 0
    #simplified[simplified < 2] = 0

    #plt.imshow(simplified)
    #plt.show()

    #print("unique values simplified:", np.unique(simplified))


    #plt.imshow(simplified)
    #plt.show()

    # normalize simplified to 0-1
    depth_norm = simplified / np.max(simplified)

    #print("atts_norm", atts_norm.min(), atts_norm.max())
    #print("depth_norm", depth_norm.min(), depth_norm.max())

    # plt grid 1x2
    #fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # plot the attention map
    #axs[0].imshow(atts_norm)
    #axs[0].set_title('Attention Map')

    # plot the depth map
    #axs[1].imshow(depth_norm)
    #axs[1].set_title('Depth Map')

    #plt.show()

    ############################
    # FINAL ATT * GLOBAL DEPTH #
    ############################

    w_attention, w_depth = calculate_weights(atts_norm, depth_norm)
    att_depth = (atts_norm * w_attention) * (depth_norm * w_depth)

    # normalize simplified from 0 to 255
    att_depth = cv2.normalize(att_depth, None, 0, 255, cv2.NORM_MINMAX)


    #plt.imshow(att_depth)
    #plt.show()

    # Assuming att_depth is your depth data array
    bins, hist = create_dynamic_depth_bins(att_depth, overlap=0.2, peak_prominence=0.05, bin_width=0.1, bins=100)

    #print("Bins:", bins)

    #plt.hist(att_depth.flatten(), bins=100)
    #plt.show()


    ########################
    # THRESHOLDED & RESULT #
    ########################


    #print(att_depth.min(), att_depth.mean(), att_depth.max(), att_depth.std())

    att_depth_masked = att_depth.copy()

    #threshold = att_depth.mean()
    #threshold = att_depth.std()
    threshold = (att_depth.mean() + att_depth.std()) / 2

    att_depth_masked[att_depth_masked <= threshold] = 0
    att_depth_masked[att_depth_masked > threshold] = 255

    #plt.imshow(att_depth_masked)
    #plt.show()


    final_image = image.copy()

    # convert final_image to numpy array
    final_image = np.array(final_image)
    final_image_loop = final_image.copy()


    unique_values = np.unique(simplified)

    proposed_boxes = get_boxes(att_depth_masked)




    ###################
    # DEPTH SPLITTING #
    ###################

    # let's create a mask for each unique value in simplified
    #
    # let's plt a grid with the masks
    #fig, axs = plt.subplots(2, int(len(unique_values) / 2) + 1, figsize=(20, 10))

    depth_layers_image = []

    for idx, value in enumerate(unique_values):
        # create a mask for the current value
        mask = np.zeros_like(simplified)
        mask[simplified == value] = 255

        depth_layer = mask.copy()

        mask = depth_layer * att_depth_masked

        # assign 255 to pixels with values greater than 0
        # because with the * operator, we lost the 255 normalization
        mask[mask > 0] = 255

        # Add the depth_layer_atts to the list (for visualization)
        depth_layers_image.append([depth_layer, att_depth_masked, mask])

        # plot the mask
        #grid_i = idx // (len(unique_values) // 2 + 1)  # Row index (0 or 1)
        #grid_j = idx % (len(unique_values) // 2 + 1)   # Column index

        #axs[grid_i, grid_j].imshow(mask)
        #axs[grid_i, grid_j].set_title(f"Mask {i}")

        #plt.imshow(mask)
        #plt.show()

        # let's implement morphological Dilation and Erosion to remove noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)

        # let's find the contours
        boxes = get_boxes(mask)
        filtered_boxes = non_maximum_suppression(boxes, iou_threshold=0.3)

        #for box in filtered_boxes:
        #    x, y, w, h = box
        #    color = colors[int(idx)] # cycle colors if more boxes than colors
        #    cv2.rectangle(final_image_loop, (x, y), (x+w, y+h), color, 2)

        # print(boxes)
        proposed_boxes.extend(filtered_boxes)

        # plt.imshow(mask)
        # plt.show()


    # show the grid
    #plt.show()

    # show the final image after loop
    #plt.imshow(final_image_loop)
    #plt.show()

    #print("Proposed bounding boxes:", proposed_boxes)

    # remove duplicates boxes
    filtered_boxes = non_maximum_suppression(proposed_boxes, iou_threshold=0.3)
    #print("Filtered bounding boxes:", filtered_boxes)

    # Draw the filtered boxes on the image
    for idx, box in enumerate(filtered_boxes):
        x, y, w, h = box
        color = colors[idx % len(colors)] # cycle colors if more boxes than colors
        cv2.rectangle(final_image, (x, y), (x+w, y+h), color, 2)

    # Display the final image with bounding boxes
    #plt.imshow(final_image)
    #plt.show()

    

    ###############
    # FINAL BOXES #
    ###############


    # Test with the given bounding boxes
    final_final_image = image.copy()
    final_final_image = np.array(final_final_image)
    #final_boxes = filter_bounding_boxes(filtered_boxes)
    final_boxes = filtered_boxes
    #print(final_boxes)

    for idx, box in enumerate(final_boxes):
        x, y, w, h = box
        color = colors[idx % len(colors)] # cycle colors if more boxes than colors
        cv2.rectangle(final_final_image, (x, y), (x+w, y+h), color, 2)

    #plt.imshow(final_final_image)
    #plt.show()

            
    # let´s draw the final image with the proposed_boxes matched with the predicted_boxes
    final_boxes_image = get_output_image(pil_img, final_boxes, ground_truth)


    #################
    # LOCAL RESULTS #
    #################

    ground_truth_as_list = convert_ground_truth_voc2007_to_list(ground_truth)
    local_corloc, ious = get_corloc_and_ious(ground_truth_as_list, final_boxes)

    # Storing image-corloc to obtain final dataset-corloc later
    corloc[i] = local_corloc

    # corloc_now is the mean of the corloc until now
    corloc_now = corloc[:i+1].sum() / (i+1)

    # show the image name in the progress bar without breaking the tqdm
    progress_bar.set_description(f"Processed {img_name} // Image corloc: {local_corloc:.2f} // Corloc now: {corloc_now:.2f}")


    ########################
    # PLOTTING THE PROCESS #
    ########################

    # Plot 6 images in a grid of 2x3
    #title = f"Image: {img_name} // Corloc: {corloc[i]}"
    title = f"Image: {img_name} // Corloc: {corloc[i]:.2f}"

    # Let´s draw the process intermediate images as a grid:
    plot_intermediate_images(
        title, pil_img, atts_norm, depth_norm, 
        att_depth, 
        att_depth_masked, final_image, final_final_image,
        show = False, filename = f"analysis/{timestamp}_images/{img_name}_process.png"
    )

    # Let´s draw the depth_layers_image as a grid
    plot_depth_layers(
        title, depth_layers_image,
        show = False, filename = f"analysis/{timestamp}_images/{img_name}_depth_layers.png"
    )





    # plt.imshow(final_boxes_image)
    # plt.show()

    ###################
    # STORING RESULTS #
    ###################

    # save the final image, which is a numpy.ndarray
    final_boxes_image = Image.fromarray(final_boxes_image)
    final_boxes_image.save(f"results/{timestamp}_images/{img_name}.jpg")

    # encode data as base64
    ious_b64 = base64.b64encode(json.dumps(ious).encode())
    predicted_boxes_b64 = base64.b64encode(json.dumps(final_boxes).encode())
    ground_truth_b64 = base64.b64encode(json.dumps(ground_truth_as_list).encode())

    # write the results to the file: i, img_name, corloc, ious, predicted_boxes, ground_truth
    results_file.write(f"{i},{img_name},{corloc[i]},{ious_b64},{predicted_boxes_b64},{ground_truth_b64}\n")


############################
# FINAL CORLOC CALCULATION #
############################

# Let's calculate corloc as the percentage of images with at least one IoU > 0.5
corloc = corloc.sum() / len(images)

print("-------------------")
print("Final CorLoc: ", corloc)
print("-------------------")

results_file.close()
