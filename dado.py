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

import sys
import torch
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

import cv2

import base64
import json
import argparse
from datetime import datetime

import numpy as np


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
images = ['009331']
#images = ['001258']


all_predictions_with_scores = []
total_tp = 0
total_fp = 0
total_fn = 0

# Store ground truths keyed by image name for easy lookup in AP calculation ---
ground_truths_dict = {}

corloc = np.zeros(len(images))

print("Processing images...")

# initialize tqdm with a progress bar
progress_bar = tqdm(images)

# we will use this flag to stop reading the file
# in case we reach the end of the file
continue_reading = True

# let's use tqdm to show a progress bar
for i, img_name in enumerate(progress_bar):

    # Flag to track if the image was processed in this iteration
    image_was_processed = False

    # --- Read existing results if available ---
    if continue_reading:
        line = results_file.readline()
        if line:
            parts = line.strip().split(",")
            img_name_file = parts[1]
            if img_name == img_name_file:
                # Image found in previous results, skip processing but update totals and store data
                local_corloc = float(parts[2])
                corloc[i] = local_corloc

                # --- Decode and use stored TP/FP/FN for totals ---
                try:
                    # Assuming TP, FP, FN are stored starting from the 7th column (index 6)
                    if len(parts) > 8: # Check if enough columns exist
                        stored_tp = int(parts[6])
                        stored_fp = int(parts[7])
                        stored_fn = int(parts[8])
                        total_tp += stored_tp
                        total_fp += stored_fp
                        total_fn += stored_fn
                        # Also store these for the progress bar description later
                        current_img_tp = stored_tp
                        current_img_fp = stored_fp
                        current_img_fn = stored_fn
                    else:
                        # Handle older entries without TP/FP/FN stored
                        print(f"Warning: TP/FP/FN not found for {img_name} in results file. Skipping total accumulation for this entry.")
                        current_img_tp, current_img_fp, current_img_fn = -1, -1, -1 # Use indicators

                    # --- Also load stored ground truth and predicted boxes with scores ---
                    ground_truth_as_list = json.loads(base64.b64decode(parts[5]).decode('utf-8'))
                    ground_truths_dict[img_name] = ground_truth_as_list # Store GT

                    predicted_boxes_data = json.loads(base64.b64decode(parts[4]).decode('utf-8'))

                    # Add stored predictions to the global list with image name
                    for box_data in predicted_boxes_data:
                         if len(box_data) == 5: # Check if score is present ([x, y, w, h, score])
                             all_predictions_with_scores.append([img_name] + box_data)
                         elif len(box_data) == 4: # Handle older entries without score ([x, y, w, h])
                             print(f"Warning: Score not found for box in {img_name} from results file. Assigning dummy score 0 for AP calculation.")
                             all_predictions_with_scores.append([img_name] + box_data + [0.0])
                         else:
                             print(f"Warning: Unexpected box data format for {img_name} from results file: {box_data}. Skipping.")


                except (IndexError, ValueError, json.JSONDecodeError) as e:
                    print(f"Warning: Error reading/decoding results for {img_name}: {e}. Skipping total accumulation and data loading for this entry.")
                    current_img_tp, current_img_fp, current_img_fn = -2, -2, -2 # Use indicators


                corloc_now = corloc[:i+1].sum() / (i+1)
                progress_bar.set_description(f"Processed {img_name} // Image corloc: {local_corloc:.2f} // TP: {current_img_tp}, FP: {current_img_fp}, FN: {current_img_fn} // Corloc now: {corloc_now:.2f}")

                # --- SKIP the rest of the loop for this image ---
                continue
        else:
            continue_reading = False

    # --- If the image was NOT skipped, process it ---
    # This block contains all the code that defines atts_norm, depth_norm,
    # att_depth_unmasked, att_depth_masked, final_image, final_final_image,
    # depth_layers_image, final_boxes, ground_truth_as_list, scored_predictions_this_image,
    # img_tp, img_fp, img_fn, local_corloc, ious, etc.
    # These variables WILL be defined if this block runs.

    image_was_processed = True # Set flag since we are processing

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
    ground_truth_as_list = convert_ground_truth_voc2007_to_list(ground_truth)
    # --- Store ground truth in dictionary ---
    ground_truths_dict[img_name] = ground_truth_as_list
    # ----------------------------------------

    plt.imshow(ground_truth_img)
    plt.show()


    #########################
    # GETTING THE ATTENTION #
    #########################
    patch_size = params['patch_size']
    img_paded = img_tensor_padded(img_tensor, patch_size).to(device)
    model = load_dino1_model(patch_size, params['ssl_checkpoint'], device, params['img_size'])
    model.to(device)
    model.eval()
    atts = get_attentions(model, img_paded, patch_size)
    sum_atts = atts.max(0)
    sum_atts = cv2.resize(sum_atts, (pil_img.size[0], pil_img.size[1]))
    atts_norm = sum_atts / np.max(sum_atts)

    #################
    # GETTING DEPTH #
    #################
    image = load_image(image_path)
    model_depth, feature_extractor = load_model()
    depth_map = get_depth_map(image, model_depth, feature_extractor)
    depth_map = cv2.resize(depth_map, (image.size[0], image.size[1]))

    height, width = depth_map.shape
    start_row = int(height * 0.75)
    scaling_factor = np.linspace(1, 0.5, height - start_row)
    scaling_mask = np.ones((height, width))
    for idx, factor in enumerate(scaling_factor):
        scaling_mask[start_row + idx, :] = factor
    depth_map_normalized_0_1 = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    modified_depth_map = depth_map_normalized_0_1 * scaling_mask
    modified_depth_map = np.clip(modified_depth_map, 0, np.max(depth_map_normalized_0_1)).astype(depth_map_normalized_0_1.dtype)
    modified_depth_map_0_1 = modified_depth_map

    depth_map_0_255 = cv2.normalize(modified_depth_map, None, 0, 255, cv2.NORM_MINMAX)

    segmented_map = depth_map_0_255.copy()
    segmented_map = cv2.normalize(segmented_map, None, 0, 9, cv2.NORM_MINMAX).astype(np.uint8)
    simplified = segmented_map.copy()

    depth_norm = modified_depth_map_0_1

    ############################
    # FINAL ATT * GLOBAL DEPTH #
    ############################
    w_attention, w_depth = calculate_weights(atts_norm, depth_norm)
    att_depth_unmasked = (atts_norm * w_attention) * (depth_norm * w_depth)

    att_depth_for_masking = cv2.normalize(att_depth_unmasked, None, 0, 255, cv2.NORM_MINMAX)


    ########################
    # THRESHOLDED & RESULT #
    ########################
    att_depth_masked = att_depth_for_masking.copy()
    threshold = (att_depth_for_masking.mean() + att_depth_for_masking.std()) / 2
    att_depth_masked[att_depth_masked <= threshold] = 0
    att_depth_masked[att_depth_masked > threshold] = 255

    # --- Define final_image here before its first use ---
    final_image = image.copy()
    final_image = np.array(final_image)
    # final_image_loop is also defined here if you use it later
    final_image_loop = final_image.copy() # Keep if needed for drawing intermediate boxes
    # -----------------------------------------------------


    proposed_boxes = get_boxes(att_depth_masked)


    ###################
    # DEPTH SPLITTING #
    ###################
    unique_values = np.unique(simplified)
    depth_layers_image = []
    for idx, value in enumerate(unique_values):
        mask = np.zeros_like(simplified)
        mask[simplified == value] = 255
        depth_layer = mask.copy()
        mask = depth_layer * att_depth_masked
        mask[mask > 0] = 255
        depth_layers_image.append([depth_layer, att_depth_masked, mask])
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        boxes = get_boxes(mask)
        filtered_boxes = non_maximum_suppression(boxes, iou_threshold=0.3)
        proposed_boxes.extend(filtered_boxes)

    filtered_boxes = non_maximum_suppression(proposed_boxes, iou_threshold=0.3)
    final_boxes = filtered_boxes # These are the final box coordinates [x, y, w, h]


    # Draw the filtered boxes on final_image - This block should be here AFTER final_image is defined
    for idx, box in enumerate(filtered_boxes):
        x, y, w, h = box
        color = colors[idx % len(colors)] # cycle colors if more boxes than colors
        cv2.rectangle(final_image, (x, y), (x+w, y+h), color, 2)


    # --- Calculate proxy scores for final_boxes and add to global list ---
    scored_predictions_this_image = []
    # Ensure att_depth_unmasked is a NumPy array
    att_depth_np = np.array(att_depth_unmasked)

    for box in final_boxes:
        x, y, w, h = box
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(att_depth_np.shape[1], x + w)
        y2 = min(att_depth_np.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
             score = 0.0
             print(f"Warning: Invalid box after clipping for {img_name}: {box}. Assigning score 0.")
        else:
            region = att_depth_np[y1:y2, x1:x2]
            if region.size > 0:
                score = float(np.max(region))
            else:
                score = 0.0

        scored_predictions_this_image.append([x, y, w, h, score])
        all_predictions_with_scores.append([img_name, x, y, w, h, score])
    # -------------------------------------------------------------------

    ###############
    # FINAL BOXES #
    ###############

    # Test with the given bounding boxes - Define final_final_image here
    final_final_image = image.copy()
    final_final_image = np.array(final_final_image)

    for idx, box in enumerate(final_boxes):
        x, y, w, h = box
        color = colors[idx % len(colors)]
        cv2.rectangle(final_final_image, (x, y), (x+w, y+h), color, 2)

    # final_boxes_image is created later for saving, using get_output_image
    plt.imshow(final_final_image)
    plt.show()




    #################
    # LOCAL RESULTS #
    #################

    # --- Calculate TP, FP, FN for the current image (using IoU=0.5) ---
    # This block should be here since image_was_processed is True
    img_tp = 0
    img_fp = 0
    img_fn = 0

    matched_gt = [False] * len(ground_truth_as_list)
    matched_pred = [False] * len(final_boxes)

    for pred_idx, pred_box in enumerate(final_boxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth_as_list):
            iou = bbox_iou(pred_box, gt_box)
            if iou >= 0.5 and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1 and not matched_gt[best_gt_idx]:
            img_tp += 1
            matched_gt[best_gt_idx] = True
            matched_pred[pred_idx] = True

    for pred_idx, matched in enumerate(matched_pred):
        if not matched:
            img_fp += 1

    for gt_idx, matched in enumerate(matched_gt):
        if not matched:
            img_fn += 1

    # Add to total counts
    total_tp += img_tp
    total_fp += img_fp
    total_fn += img_fn

    # Also store these for the progress bar description and file writing
    current_img_tp = img_tp
    current_img_fp = img_fp
    current_img_fn = img_fn
    # ------------------------------------------------------------------

    local_corloc, ious = get_corloc_and_ious(ground_truth_as_list, final_boxes)

    corloc[i] = local_corloc
    corloc_now = corloc[:i+1].sum() / (i+1)

    # Update progress bar description with calculated TP/FP/FN
    progress_bar.set_description(f"Processed {img_name} // Image corloc: {local_corloc:.2f} // TP: {current_img_tp}, FP: {current_img_fp}, FN: {current_img_fn} // Corloc now: {corloc_now:.2f}")


    # --- Plotting and Saving (Only run if processed - which is true here) ---
    # The previous 'if image_was_processed' check is redundant if this block
    # is placed here after all processing and before the loop ends.
    # If you prefer explicitness, you can keep 'if image_was_processed:' wrapper.

    # Define title
    title = f"Image: {img_name} // Corloc: {corloc[i]:.2f} // TP: {current_img_tp}, FP: {current_img_fp}, FN: {current_img_fn}"

    # Plotting
    plot_intermediate_images(
        title, pil_img, atts_norm, depth_norm,
        att_depth_unmasked,
        att_depth_masked, final_image, final_final_image, # final_image and final_final_image are defined above
        show = False, filename = f"analysis/{timestamp}_images/{img_name}_process.png"
    )

    plot_depth_layers(
        title, depth_layers_image, # depth_layers_image is defined above
        show = False, filename = f"analysis/{timestamp}_images/{img_name}_depth_layers.png"
    )

    # Saving final image
    final_boxes_image = Image.fromarray(get_output_image(pil_img, final_boxes, ground_truth)) # Uses final_boxes, ground_truth defined above
    final_boxes_image.save(f"results/{timestamp}_images/{img_name}.jpg")

    # Writing to results file
    ious_b64 = base64.b64encode(json.dumps(ious).encode()).decode('utf-8')
    # Store predicted boxes *with scores*
    predicted_boxes_b64 = base64.b64encode(json.dumps(scored_predictions_this_image).encode()).decode('utf-8') # Uses scored_predictions_this_image defined above
    ground_truth_b64 = base64.b64encode(json.dumps(ground_truth_as_list).encode()).decode('utf-8') # Uses ground_truth_as_list defined above

    results_file.write(f"{i},{img_name},{corloc[i]},{ious_b64},{predicted_boxes_b64},{ground_truth_b64},{current_img_tp},{current_img_fp},{current_img_fn}\n")


# ... (Rest of the code, including compute_ap_pascal and final prints) ...


############################
# FINAL METRICS CALCULATION #
############################

# Let's calculate corloc as the percentage of images with at least one IoU > 0.5
corloc_dataset = corloc.sum() / len(images) # Renamed to avoid conflict

# --- Calculate overall Precision and Recall at IoU=0.5 ---
overall_precision_at_0_5_iou = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_recall_at_0_5_iou = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
# ----------------------------------------------------------

# --- Implement/Call the AP50 calculation ---

def compute_ap_pascal(all_preds, ground_truths_dict, iou_threshold=0.5):
    """
    Computes Average Precision (AP) for a given IoU threshold using Pascal VOC logic.

    Args:
        all_preds (list): List of all predictions across dataset:
                          [[img_name, x, y, w, h, score], ...]
        ground_truths_dict (dict): Dictionary mapping img_name to list of GT boxes:
                                   {img_name: [[x, y, w, h], ...], ...}
        iou_threshold (float): The IoU threshold for considering a match (e.g., 0.5 for AP50).

    Returns:
        float: The Average Precision at the given IoU threshold.
    """
    # Sort predictions by confidence score in descending order
    all_preds.sort(key=lambda x: x[5], reverse=True)

    n_ground_truth = sum(len(gt) for gt in ground_truths_dict.values()) # Total number of ground truth objects
    if n_ground_truth == 0:
        return 0.0 # AP is 0 if there are no ground truth objects

    # Keep track of matched ground truth boxes for each image
    # Use a set or list of booleans per image
    matched_gt_per_image = {img_name: [False] * len(gt_boxes) for img_name, gt_boxes in ground_truths_dict.items()}

    tps = [] # List to store whether each prediction is a TP (1) or FP (0)
    fps = []

    # Iterate through sorted predictions
    for img_name, pred_x, pred_y, pred_w, pred_h, score in all_preds:
        pred_box = [pred_x, pred_y, pred_w, pred_h]
        gt_boxes = ground_truths_dict.get(img_name, [])
        gt_matched_flags = matched_gt_per_image.get(img_name, [])

        best_iou = 0
        best_gt_idx = -1

        # Find best ground truth match for this prediction
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = bbox_iou(pred_box, gt_box) # Ensure bbox_iou is accessible and works with [x,y,w,h]
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx != -1 and not gt_matched_flags[best_gt_idx]:
            # This prediction is a True Positive
            tps.append(1)
            fps.append(0)
            gt_matched_flags[best_gt_idx] = True # Mark the ground truth as matched
        else:
            # This prediction is a False Positive
            tps.append(0)
            fps.append(1)

    # Calculate cumulative TP and FP
    cumulative_tps = np.cumsum(tps)
    cumulative_fps = np.cumsum(fps)

    # Calculate precision and recall at each step
    precisions = cumulative_tps / (cumulative_tps + cumulative_fps)
    recalls = cumulative_tps / n_ground_truth

    # Compute AP using the 11-point interpolation method (Pascal VOC 2007)
    # Or the more modern method (Pascal VOC 2010-2012 / COCO - Area under curve)
    # Let's use the AUC method for simplicity as it's more common now.

    # Append (0, 0) and (1, 0) to precision-recall points for correct integration
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # Interpolate precision values to make the curve monotonic
    # For each recall level r', set the precision to the maximum precision
    # found for any recall r >= r'.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i+1])

    # Compute the area under the PR curve using the trapezoidal rule
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])

    return ap


# --- Call the AP50 calculation after the loop ---
ap50 = compute_ap_pascal(all_predictions_with_scores, ground_truths_dict, iou_threshold=0.5)
# -----------------------------------------------


print("-------------------")
print(f"Final CorLoc: {corloc_dataset:.2f}")
print(f"Overall Precision @ IoU=0.5: {overall_precision_at_0_5_iou:.2f}")
print(f"Overall Recall @ IoU=0.5: {overall_recall_at_0_5_iou:.2f}")
print(f"Average Precision @ IoU=0.5 (AP50 - Proxy Score): {ap50:.2f}")
print(f"Total TP (IoU=0.5): {total_tp}")
print(f"Total FP (IoU=0.5): {total_fp}")
print(f"Total FN (IoU=0.5): {total_fn}")
print("-------------------")

results_file.close()
