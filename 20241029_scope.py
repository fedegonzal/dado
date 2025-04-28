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

from transformers import  DPTForDepthEstimation, DPTImageProcessor

import base64
import json
import argparse
from datetime import datetime




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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# let's create or open a file to store the results, we want to read and write

parser = argparse.ArgumentParser()
parser.add_argument('--timestamp', type=str, help='prefix results file to store the results')
args = parser.parse_args()

# timestamp has the format: YYYYMMDD, else today
timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d")

results_filename = f"results/{timestamp}_results.csv"

# create a folder to store the images if it doesn't exist
if not os.path.exists(f"results/{timestamp}_images"):
    os.makedirs(f"results/{timestamp}_images")

# the file where we store the results, create if it doesn't exist
if not os.path.exists(results_filename):
    results_file = open(results_filename, "w+")
else:
    results_file = open(results_filename, "r+")


# Get N random images from dataset_path
# n=None to get all images
dataset_path = "datasets/VOC2007/VOCdevkit/VOC2007"
images = get_images(dataset_path, -1)

#images = ['005881', '000250', '001855', '007586', '009079', '003159', '009745', '004801', '004244', '006185', '002405', '000334', '008349', '007853', '001263', '007193', '008856', '003678', '008833', '000109', '002873', '009881', '007040', '007751', '008917', '003369', '003629', '004828', '004192', '008415', '008733', '009418', '006694', '005676', '006339', '001590', '004281', '008096', '009438', '006212', '005064', '008372', '009343', '001185', '009307', '003433', '002266', '009214', '009421', '007417', '000951', '004359', '005732', '007697', '001981', '003147', '002224', '003051', '004706', '000047', '001393', '001510', '002977', '003994', '002880', '005790', '006247', '006282', '004441', '006822', '006626', '007398', '007261', '009246', '007503', '001343', '001250', '003807', '000667', '003120', '000489', '007372', '001455', '000470', '004797', '005101', '000917', '005312', '007847', '009904', '006425', '000483', '001927', '009331', '004518', '000173', '002439', '005365', '007621', '006968']
#images = ['009331', '002439', '005365', '007621', '006968']
#images = ['009331']
#images = ['001258']

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

    image_path = f'{dataset_path}/JPEGImages/{img_name}.jpg'
    annotation_path = f'{dataset_path}/Annotations/{img_name}.xml'


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

    #noise = estimate_noise(sum_atts)
    #entropy = calculate_entropy(sum_atts)

    # Resize the attentions to the original image size
    sum_atts_resized = cv2.resize(sum_atts, (pil_img.width, pil_img.height))

    #plt.imshow(sum_atts_resized)
    #plt.show()

    # Normalize sum_atts_resized from 0 to 1
    # sum_atts_resized_norm = (sum_atts_resized - np.min(sum_atts_resized)) / (np.max(sum_atts_resized) - np.min(sum_atts_resized))

    #print("sum_atts_resized_norm", sum_atts_resized_norm.min(), sum_atts_resized_norm.max())
    #plt.imshow(sum_atts_resized_norm)
    #plt.show()


    #################
    # GETTING DEPTH #
    #################

    # Get the depth prediction
    depth_feats, depth_image = get_depth_prediction(pil_img, params['depth_checkpoint'])

    # Obtain the depth map

    # Resize depth_image to the original image size
    numpy_depth_image = depth_image.squeeze(0).cpu().numpy()
    depth_image_resized = cv2.resize(numpy_depth_image, pil_img.size)

    # plt.imshow(depth_image_resized)
    # plt.show()

    # Normalizing depth
    # depth_image_resized_norm = (depth_image_resized - np.min(depth_image_resized)) / (np.max(depth_image_resized) - np.min(depth_image_resized))


    ############################
    # FINAL ATT * GLOBAL DEPTH #
    ############################

    # Normalize maps for fair comparison
    sum_atts_resized_norm = (sum_atts_resized - sum_atts_resized.mean()) / sum_atts_resized.std()
    depth_image_resized_norm = (depth_image_resized - depth_image_resized.mean()) / depth_image_resized.std()

    w_attention, w_depth = calculate_weights(sum_atts_resized_norm, depth_image_resized_norm)

    # Compute the final attention map    
    #final_attention_map_ok = (w_attention * sum_atts_resized_norm) * (w_depth * depth_image_resized_norm)    
    #final_attention_map_simple = (sum_atts_resized_norm) * (depth_image_resized_norm)
    final_attention_map = burned_image = color_burn(sum_atts_resized_norm, depth_image_resized_norm)


    ########################
    # THRESHOLDED & RESULT #
    ########################

    # Apply a binary threshold to the image
    _, final_att_thresholded = cv2.threshold(final_attention_map, final_attention_map.std(), final_attention_map.max(), cv2.THRESH_BINARY)

    # Get the predicted boxes from the final image using contours
    predicted_boxes = get_boxes(final_att_thresholded)

    final_image_simple = get_output_image(pil_img, predicted_boxes, ground_truth)

    final_boxes = predicted_boxes

    ###################
    # DEPTH SPLITTING #
    ###################

    # Create overlapping depth bins
    overlapping_bins = create_overlapping_depth_bins(depth_image_resized_norm, num_bins=7, overlap=0.2)

    # Segment depth map into bins
    segmented_depth_map = segment_depth_map(depth_image_resized_norm, overlapping_bins)

    # Let's separate the segmented_depth_map into layers
    depth_layers = []

    for idx_bin, bin in enumerate(overlapping_bins):
        mask = (segmented_depth_map == idx_bin+1) # because values start at 1
        depth_layer = np.where(mask, segmented_depth_map, 0)

        # let's normalize the depth_layer from 0 to 1
        depth_layer = (depth_layer - np.min(depth_layer)) / (np.max(depth_layer) - np.min(depth_layer))

        depth_layers.append(depth_layer)


    proposed_boxes = []

    depth_layers_image = []

    for depth_layer in depth_layers:

        #resize depth_layer to att size
        depth_layer = cv2.resize(depth_layer, (sum_atts_resized_norm.shape[1], sum_atts_resized_norm.shape[0]))

        # normalize depth_layer from 0 to 1
        depth_layer = (depth_layer - np.min(depth_layer)) / (np.max(depth_layer) - np.min(depth_layer))

        # Here I should use the weights calculated before, but locally
        # multiply depth_layer by att
        depth_layer_atts = (w_attention * sum_atts_resized_norm) * (w_depth * depth_layer)
        #depth_layer_atts = color_burn(sum_atts_resized_norm, depth_layer)

        # normalize depth_layer_atts from 0 to 1
        depth_layer_atts = (depth_layer_atts - np.min(depth_layer_atts)) / (np.max(depth_layer_atts) - np.min(depth_layer_atts))

        #plt.imshow(depth_layer_atts)
        #plt.show()

        # Apply a binary threshold to the image
        _, thresholded_layer = cv2.threshold(depth_layer_atts, depth_layer_atts.std(), depth_layer_atts.max(), cv2.THRESH_BINARY)

        # Add the depth_layer_atts to the list (for visualization)
        depth_layers_image.append([depth_layer, depth_layer_atts, thresholded_layer])

        # thresholded_layer width * height
        depth_layer_atts_size = depth_layer_atts.shape[0] * depth_layer_atts.shape[1]

        # count pixels=1 in thresholded_layer
        depth_layer_atts_count = cv2.countNonZero(thresholded_layer)

        # high value pixels proportion
        depth_layer_atts_proportion = depth_layer_atts_count / depth_layer_atts_size

        # if the proportion is greater than 0.8, we consider it part of background and don't process it
        if depth_layer_atts_proportion > 0.8:
            layer_predicted_boxes = []
        else:
            # Get the predicted boxes from the final image using contours
            layer_predicted_boxes = get_boxes(thresholded_layer)

        # Append the predicted boxes to the proposed_boxes
        proposed_boxes.extend(layer_predicted_boxes)

        # Get the final image with the boxes drawn (predicted and ground truth)
        # final_image = get_output_image(pil_img, layer_predicted_boxes, ground_truth)

    

    ###############
    # FINAL BOXES #
    ###############

    # Let´s validate the proposed_boxes with predicted_boxes
    # if the IoU is greater than iou_threshold, we consider it a match

    iou_threshold = 0.1 # slower values, force to find new boxes
    final_boxes = []

    final_boxes = predicted_boxes

    for box in proposed_boxes:

        try:
            #final_boxes.append(proposed_box)
            ious = bbox_iou(torch.tensor(box), torch.tensor(predicted_boxes), x1y1x2y2=False)

            for iou in ious:
                # it's a really new box
                if iou < iou_threshold and iou > 0:
                    final_boxes.append(box)
        
        except:
            pass

    # remove boxes inside other boxes
    #final_boxes = remove_inside_boxes(final_boxes)
    
    # just to preserve both lists
    #final_boxes.extend(predicted_boxes)
    #final_boxes = predicted_boxes
            
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
        title, pil_img, sum_atts_resized, depth_image_resized, 
        final_attention_map, 
        final_att_thresholded, final_image_simple, final_boxes_image,
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
    predicted_boxes_b64 = base64.b64encode(json.dumps(predicted_boxes).encode())
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
