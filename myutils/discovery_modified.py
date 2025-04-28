import torch
import torch.nn as nn
import numpy as np
import cv2

from myutils.datasets import bbox_iou

# Given an Attention model (DINO) and an image,
# returns its attention matrix heatmap
def get_attentions(model, img_paded, patch_size):

    # Size for transformers
    w_featmap = img_paded.shape[-2] // patch_size
    h_featmap = img_paded.shape[-1] // patch_size


    with torch.no_grad():
        attentions = model.get_last_selfattention(img_paded[None, :, :, :])

    # we keep only the output patch attention
    # for every patch
    nh = attentions.shape[1]  # Number of heads
    atts = attentions[0, :, 0, 1:].reshape(nh, -1)

    atts = atts.reshape(nh, w_featmap, h_featmap)

    # resize to image size
    atts = nn.functional.interpolate(atts.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

    return atts



def get_boxes(final):

    # Find the contours
    contours, hierarchy = cv2.findContours(final.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    colors = [
        (0, 0, 255), # blue
        (0, 255, 0), # green
        (255, 0, 0), # red
        (255, 255, 0), # cyan
        (255, 0, 255), # magenta
        (0, 255, 255), # yellow
    ]

    the_contours = []

    # Draw the contours with different colors
    for i, contour in enumerate(contours):
        
        # Using hierarchy to filter out the inner contours
        if hierarchy[0][i][3] != -1:
            continue

        # if the contour is too small, ignore it
        if cv2.contourArea(contour) < 1000:
            continue

        # Approximate the contour with a rectangle
        x, y, w, h = cv2.boundingRect(contour)    
        the_contours.append([x, y, w, h])

    return the_contours
    


def get_output_image(pil_img, contours, ground_truth):

    img = np.array(pil_img)

    #img3 = draw_boxes(contours, img)

    # Let's define tensors for the contours and ground truth boxes with shape (n, 4)
    contours_tensor = torch.tensor([]).reshape(0, 4)
    ground_truth_tensor = torch.tensor([]).reshape(0, 4)

    # Draw the matched boxes for countours
    for i, contour in enumerate(contours):
        color = (0, 0, 255)
        thickness = 3
        x, y, w, h = contour
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        # print the contour coordinates over the image
        cv2.putText(img, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # let's append the contour to the tensor
        contours_tensor = torch.cat((contours_tensor, torch.tensor([[x, y, x + w, y + h]])))

    # Draw the matched boxes for ground truth
    for i, gt in enumerate(ground_truth):
        color = (0, 255, 0)
        thickness = 3
        bndbox = gt.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        ground_truth_tensor = torch.cat((ground_truth_tensor, torch.tensor([[xmin, ymin, xmax, ymax]])))

    return img


def estimate_noise(image):

    # let's normalize the image from 0 to 255
    image = image - np.min(image)
    image = image / np.max(image) * 255

    # convert to uint8
    image = image.astype(np.uint8)

    h, w = image.shape
    mean = np.mean(image)
    std_dev = np.std(image)
    return std_dev


from scipy.stats import entropy

def calculate_entropy(image):

    # let's normalize the image from 0 to 255
    image = image - np.min(image)
    image = image / np.max(image) * 255

    # convert to uint8
    image = image.astype(np.uint8)

    # Calculate histogram
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    
    # Normalize histogram
    hist_normalized = hist / hist.sum()

    # Calculate entropy
    try:
        ent = entropy(hist_normalized, base=2)
    except:
        ent = 0

    return ent


# Make sure bbox_iou is imported or defined before this function
# from .utils import bbox_iou # If bbox_iou is in the same package's utils file
# or from .datasets import bbox_iou # If bbox_iou is in the same package's datasets file
# Or if bbox_iou is defined in the same script, no import needed

def get_corloc_and_ious(ground_truths, predictions):
    """
    Calculates image-level CorLoc and the maximum IoU for each ground truth box.

    CorLoc for a single image is 1 if at least one predicted box has >= 0.5 IoU
    with any ground truth box, and 0 otherwise.

    Args:
        ground_truths (list): A list of ground truth bounding boxes for an image,
                              e.g., [[x, y, w, h], ...].
        predictions (list): A list of predicted bounding boxes for an image,
                            e.g., [[x, y, w, h], ...].

    Returns:
        tuple: A tuple containing:
            - local_corloc (int): 1 if the image is correctly localized, 0 otherwise.
            - max_ious_for_gt (list): A list where each element is the maximum IoU
                                      found for the corresponding ground truth box
                                      against all predicted boxes.
    """
    local_corloc = 0
    max_ious_for_gt = []
    iou_threshold = 0.5 # CorLoc uses a 0.5 IoU threshold

    # Handle cases with no ground truths or no predictions
    if not ground_truths:
        # If no ground truth, we can't find any TP, CorLoc is 0, no meaningful IoUs
        return 0, []
    if not predictions:
        # If no predictions, no ground truths can be matched
        return 0, [0.0] * len(ground_truths) # Max IoU for each GT is 0

    # Keep track of whether any GT was matched with sufficient IoU
    any_gt_matched_sufficiently = False

    # Iterate through each ground truth box
    for gt_box in ground_truths:
        max_iou_with_preds = 0.0

        # Find the maximum IoU between the current ground truth box and all predictions
        for pred_box in predictions:
            iou = bbox_iou(gt_box, pred_box) # Use the bbox_iou function
            max_iou_with_preds = max(max_iou_with_preds, iou)

        # Append the maximum IoU found for this ground truth box
        max_ious_for_gt.append(max_iou_with_preds)

        # Check if this ground truth was matched with at least IoU >= 0.5
        if max_iou_with_preds >= iou_threshold:
            any_gt_matched_sufficiently = True

    # Determine local_corloc based on if *any* ground truth was sufficiently matched
    if any_gt_matched_sufficiently:
        local_corloc = 1

    return local_corloc, max_ious_for_gt



def get_corloc_and_ious__(ground_truth_as_list, predicted_boxes):

    corloc = 0
    ious = []
    
    for pred_box in predicted_boxes:
        pred_box = [pred_box[0], pred_box[1], pred_box[0]+pred_box[2], pred_box[1]+pred_box[3]]

        iou = bbox_iou(torch.tensor(pred_box), torch.tensor(ground_truth_as_list))

        #print("IoU: ", iou)

        if iou.max().item() > 0.5:
            corloc = 1

        ious.append(iou)

    #print(f"Image {i} of {len(images)-1} // {img_name} // CorLoc: {corloc[i]} // Partial CorLoc: {corloc[:i+1].sum() / (i+1)}")
    
    # ious is a list of tensors, we need to convert it to a list of lists
    ious = [iou.tolist() for iou in ious]

    return corloc, ious


import torch

def odap50_no_score(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    gt_boxes: list of [x1, y1, x2, y2]
    pred_boxes: list of [x1, y1, x2, y2]
    """
    if len(gt_boxes) == 0:
        return 0.0

    pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

    if len(pred_tensor) == 0:
        return 0.0

    matched_gt = set()
    true_positives = 0

    for pred in pred_tensor:
        ious = bbox_iou(pred, gt_tensor.T)  # shape: (num_gts,)
        best_iou, best_gt_idx = torch.max(ious, dim=0)
        if best_iou >= iou_threshold and best_gt_idx.item() not in matched_gt:
            true_positives += 1
            matched_gt.add(best_gt_idx.item())

    recall = true_positives / len(gt_tensor)
    return recall  # This is effectively odAP50 in the no-score setting
