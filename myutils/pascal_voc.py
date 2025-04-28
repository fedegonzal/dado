"""Module providing a function printing python version."""

import xml.etree.ElementTree as ET
import cv2
import numpy as np


# Read an image and its annotations from Pascal VOC 2007
# Returns its ground_truth, ground_truth_img and the image with boxes
def get_ground_truth_voc2007(xml_path, pil_img):

    # Load the xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ground_truth = root.findall('object')

    ground_truth_img = draw_ground_truth_voc2007(pil_img, ground_truth)

    return ground_truth, ground_truth_img


# Given a Pascal VOC ground truth, draw boxes over a imagen
def draw_ground_truth_voc2007(pil_img, ground_truth):

    img = np.array(pil_img)

    # Draw the contours of the objects
    for obj in ground_truth:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return img


# Given a List of boxes, draw them over a imagen
def draw_boxes(pil_img, boxes, color=(0, 255, 0)):

    img = np.array(pil_img)

    # Draw the contours of the objects
    for box in boxes:
        xmin, ymin, xmax, ymax = box

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

    return img


# Convert a Pascal VOC 2007 ground truth, to list format
# [ [xmin, ymin, xmax, ymax] , ... ]
def convert_ground_truth_voc2007_to_list(ground_truth):

    ground_truth_list = []
    for obj in ground_truth:
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        ground_truth_list.append([xmin, ymin, xmax, ymax])

    return ground_truth_list



def get_images(dataset_path, n):
    """ Function printing python version. """

    import os
    import random

    images = []

    for dirpath, dirnames, filenames in os.walk(f'{dataset_path}/JPEGImages'):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            images.append(filename.split('.')[0])

    if n < 0:
        return images
    else:
        return random.sample(images, n)
