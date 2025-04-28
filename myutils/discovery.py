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



def get_corloc_and_ious(ground_truth_as_list, predicted_boxes):

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
