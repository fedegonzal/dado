import os
import torch
import json
import torchvision
import numpy as np
import skimage.io
import pdb
import pandas as pd
import pickle

from PIL import Image
from tqdm import tqdm
from torchvision import transforms as pth_transforms
from torch.utils.data import Dataset

# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class ImagenetDataset(Dataset):
    def __init__(self, root_path, image_path, transform=None):
        self.image_path = image_path
        self.root_path = root_path
        self.transform=transform
        self.images = pd.read_csv(os.path.join(self.root_path,'imagenet_images.csv'),
                    header=None)[0].unique().tolist()
        if not self.images:
            raise Exception("Path to images {} invalid!!".format(image_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_n = self.images[idx]
        img = Image.open(os.path.join(self.image_path, im_n + '.JPEG')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, im_n



class ECSSDDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform=transform
        self.images = os.listdir(image_path)
        if not self.images:
            raise Exception("Path to images {} invalid!!".format(image_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_n = self.images[idx]
        img = Image.open(os.path.join(self.image_path, im_n)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, im_n


class CUBDataset(Dataset):
    def __init__(self, image_path, dataset_set='train', transform=None):
        self.image_path = os.path.join(image_path, 'images')
        self.image_list_file = os.path.join(image_path, 'images.txt')
        self.images = pd.read_csv(self.image_list_file, header=None, sep=" ", names=['id', 'path'])
        self.split = pd.read_csv(os.path.join(image_path,
                                              'train_test_split.txt'),
                                 header=None, sep=" ", names=['id', 'split'])

        self.images = self.images.merge(self.split, on="id")
        split = 1 if dataset_set=='train' else 0
        self.images = self.images[self.images.split == split]
        self.transform=transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        im_n = self.images.iloc[idx]
        img = Image.open(os.path.join(self.image_path, im_n['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, im_n['id']


class ImageDataset:
    def __init__(self, image_path):
        self.image_path = image_path
        self.name = image_path.split("/")[-1]

        # Read the image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Build a dataloader
        img = transform(img)
        self.dataloader = [[img, image_path]]

    def get_image_name(self, *args, **kwargs):
        return self.image_path.split("/")[-1].split(".")[0]

    def load_image(self, *args, **kwargs):
        return skimage.io.imread(self.image_path)

class Dataset:
    def __init__(self, dataset_name, dataset_set, remove_hards):
        """
        Build the dataloader
        """

        self.dataset_name = dataset_name
        self.set = dataset_set

        if dataset_name == "VOC07":
            self.root_path = "datasets/VOC2007"
            self.year = "2007"
        elif dataset_name == "VOC12":
            self.root_path = "datasets/VOC2012"
            self.year = "2012"
        elif dataset_name == "COCO20k":
            self.year = "2014"
            self.root_path = f"datasets/COCO/images/{dataset_set}{self.year}"
            self.sel20k = 'datasets/coco_20k_filenames.txt'
            # JSON file constructed based on COCO train2014 gt 
            self.all_annfile = "datasets/COCO/annotations/instances_train2014.json"
            self.annfile = "datasets/instances_train2014_sel20k.json"
            if not os.path.exists(self.annfile):
                select_coco_20k(self.sel20k, self.all_annfile)
        elif dataset_name == "COCO":
            self.year = "2014"
            self.root_path = f"datasets/COCO/images/{dataset_set}{self.year}"
            # JSON file constructed based on COCO train2014 gt 
            self.all_annfile = "datasets/instances_train2017.json"
            self.annfile = "datasets/instances_train2017.json"
        elif dataset_name == "COCOminival":
            self.year = 2017
            self.root_path = f"/fs/vulcan-datasets/coco/images/val2017"
            self.all_annfile = "/fs/vulcan-datasets/coco/annotations/instances_val2017.json"
            self.annfile = "/fs/vulcan-datasets/coco/annotations/instances_val2017.json"
        elif dataset_name == "ECSSD":
            self.root_path = "saliency/data/ECSSD/images/"
        elif dataset_name == "DUTS":
            self.root_path = "saliency/data/DUTS-TE/DUTS-TE-Image/"
        elif dataset_name == "DUT-OMRON":
            self.root_path = "saliency/data/DUT-OMRON/DUT-OMRON-image/"
        else:
            raise ValueError("Unknown dataset.")
        if not os.path.exists(self.root_path):
            raise ValueError("Please follow the README to setup the datasets.")

        self.name = f"{self.dataset_name}_{self.set}"

        # Build the dataloader
        if "VOC" in dataset_name:
            self.dataloader = torchvision.datasets.VOCDetection(
                self.root_path,
                year=self.year,
                image_set=self.set,
                transform=transform,
                download=False,
            )
        elif "COCO20k" == dataset_name:
            self.dataloader = torchvision.datasets.CocoDetection(
                self.root_path, annFile=self.annfile, transform=transform
            )
        elif "COCO" == dataset_name:
            self.dataloader = torchvision.datasets.CocoDetection(
                self.root_path, annFile=self.annfile, transform=transform
            )
        elif "LVIS" == dataset_name:
            self.dataloader = LVISDetection(self.root_path,
                                            annFile=self.annFile,
                                            transform=transform)
        elif "COCOminival" == dataset_name:
            self.dataloader = torchvision.datasets.CocoDetection(
                self.root_path, annFile=self.annfile, transform=transform)
        elif "ECSSD" == dataset_name:
            self.dataloader = ECSSDDataset(self.root_path, transform=transform)
        elif "DUTS" == dataset_name:
            self.dataloader = ECSSDDataset(self.root_path, transform=transform)
        elif "DUT-OMRON" == dataset_name:
            self.dataloader = ECSSDDataset(self.root_path, transform=transform)
        elif "CUB" == dataset_name:
            self.dataloader = CUBDataset(self.root_path, dataset_set=self.set, transform=transform)
        elif "Imagenet" == dataset_name:
            self.dataloader = ImagenetDataset(self.annfile, self.root_path, transform=transform)

        else:
            raise ValueError("Unknown dataset.")

        # Set hards images that are not included
        self.remove_hards = remove_hards
        self.hards = []
        if remove_hards:
            self.name += f"-nohards"
            self.hards = self.get_hards()
            print(f"Nb images discarded {len(self.hards)}")

    def load_image(self, im_name):
        """
        Load the image corresponding to the im_name
        """
        if "VOC" in self.dataset_name:
            image = skimage.io.imread(f"datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/JPEGImages/{im_name}")
        elif "COCO" in self.dataset_name:
            # im_path = self.path_20k[self.sel_20k.index(im_name)]
            filename = im_name.zfill(12)+'.jpg'
            image = skimage.io.imread(f"datasets/COCO/images/train2014/{filename}")
        elif self.dataset_name in ['ECSSD']:
            image = skimage.io.imread(f"saliency/data/ECSSD/images/{im_name}")
        elif self.dataset_name in ['DUTS']:
            image = skimage.io.imread(f"saliency/data/DUTS-TE/DUTS-TE-Image/{im_name}")
        elif self.dataset_name == 'DUT-OMRON':
            image = skimage.io.imread(f"saliency/data/DUT-OMRON/DUT-OMRON-image/{im_name}")
        elif self.dataset_name == 'CUB':
            image = skimage.io.imread(f"/fs/vulcan-datasets/CUB/CUB_200_2011/images/{im_name}")
        elif self.dataset_name == 'Imagenet':
            image = skimage.io.imread(f"/fs/vulcan-datasets/imagenet/val/{im_name}")

        else:
            raise ValueError("Unkown dataset.")
        return image

    def get_image_name(self, inp):
        """
        Return the image name
        """
        if "VOC" in self.dataset_name:
            im_name = inp["annotation"]["filename"]
        elif "COCO" in self.dataset_name:
            try:
                im_name = str(inp[0]["image_id"])
            except:
                pdb.set_trace()
        elif "LVIS" in self.dataset_name:
            # for lvis return split/img_name
            try:
                im_name = str(inp[0]["image_id"])
            except:
                pdb.set_trace()
        elif "ECSSD" in self.dataset_name:
            im_name = inp
        elif "DUTS" in self.dataset_name:
            im_name = inp
        elif "DUT-OMRON" in self.dataset_name:
            im_name = inp
        elif "CUB" in self.dataset_name:
            im_name = inp
        elif "Imagenet" in self.dataset_name:
            im_name = inp

        return im_name

    def extract_gt(self, targets, im_name):
        if "VOC" in self.dataset_name:
            return extract_gt_VOC(targets, remove_hards=self.remove_hards)
        elif "COCO" in self.dataset_name:
            return extract_gt_COCO(targets, remove_iscrowd=True)
        elif "LVIS" in self.dataset_name:
            return extract_gt_LVIS(targets, im_name)
        elif "ECSSD" in self.dataset_name:
            return extract_gt_ECSSD(targets, im_name)
        elif "DUTS" in self.dataset_name:
            return extract_gt_DUTS(targets, im_name)
        elif "DUT-OMRON" in self.dataset_name:
            return extract_gt_DUT_OMRON(targets, im_name)
        elif "CUB" in self.dataset_name:
            return extract_gt_CUB(targets, im_name)
        elif "Imagenet" in self.dataset_name:
            return extract_gt_Imagenet(targets, im_name)

        else:
            raise ValueError("Unknown dataset")

    def extract_classes(self):
        if "VOC" in self.dataset_name:
            cls_path = f"classes_{self.set}_{self.year}.txt"
        elif "COCO" in self.dataset_name:
            cls_path = f"classes_{self.dataset}_{self.set}_{self.year}.txt"

        # Load if exists
        if os.path.exists(cls_path):
            all_classes = []
            with open(cls_path, "r") as f:
                for line in f:
                    all_classes.append(line.strip())
        else:
            print("Extract all classes from the dataset")
            if "VOC" in self.dataset_name:
                all_classes = self.extract_classes_VOC()
            elif "COCO" in self.dataset_name:
                all_classes = self.extract_classes_COCO()

            with open(cls_path, "w") as f:
                for s in all_classes:
                    f.write(str(s) + "\n")

        return all_classes

    def extract_classes_VOC(self):
        all_classes = []
        for im_id, inp in enumerate(tqdm(self.dataloader)):
            objects = inp[1]["annotation"]["object"]

            for o in range(len(objects)):
                if objects[o]["name"] not in all_classes:
                    all_classes.append(objects[o]["name"])

        return all_classes

    def extract_classes_COCO(self):
        all_classes = []
        for im_id, inp in enumerate(tqdm(self.dataloader)):
            objects = inp[1]

            for o in range(len(objects)):
                if objects[o]["category_id"] not in all_classes:
                    all_classes.append(objects[o]["category_id"])

        return all_classes

    def get_hards(self):
        hard_path = "datasets/hard_%s_%s_%s.txt" % (self.dataset_name, self.set, self.year)
        if os.path.exists(hard_path):
            hards = []
            with open(hard_path, "r") as f:
                for line in f:
                    hards.append(int(line.strip()))
        else:
            print("Discover hard images that should be discarded")

            if "VOC" in self.dataset_name:
                # set the hards
                hards = discard_hard_voc(self.dataloader)

            with open(hard_path, "w") as f:
                for s in hards:
                    f.write(str(s) + "\n")

        return hards


def discard_hard_voc(dataloader):
    hards = []
    for im_id, inp in enumerate(tqdm(dataloader)):
        objects = inp[1]["annotation"]["object"]
        nb_obj = len(objects)

        hard = np.zeros(nb_obj)
        for i, o in enumerate(range(nb_obj)):
            hard[i] = (
                1
                if (objects[o]["truncated"] == "1" or objects[o]["difficult"] == "1")
                else 0
            )

        # all images with only truncated or difficult objects
        if np.sum(hard) == nb_obj:
            hards.append(im_id)
    return hards

def extract_gt_ECSSD(targets, im_name):
    gt_path = "saliency/data/ECSSD/ground_truth_mask"
    im_name = im_name.split('.')[0] + '.png'
    if not os.path.isfile(os.path.join(gt_path, im_name)):
        raise Exception("Gt file {} not found".format(im_name))
    img = np.asarray(Image.open(os.path.join(gt_path, im_name)).convert('L'))
    if np.unique(img).shape[0] > 2:
        img[img > 0] = 255
    return img, None

def extract_gt_DUTS(targets, im_name):
    gt_path = "saliency/data/DUTS-TE/DUTS-TE-Mask/"
    im_name = im_name.split('.')[0] + '.png'
    if not os.path.isfile(os.path.join(gt_path, im_name)):
        raise Exception("Gt file {} not found".format(im_name))
    img = np.asarray(Image.open(os.path.join(gt_path, im_name)).convert('L'))
    if np.unique(img).shape[0] > 2:
        img[img > 0] = 255
    return img, None

def extract_gt_CUB(targets, im_name):
    gt_path = "/fs/vulcan-datasets/CUB/CUB_200_2011/"
    img_cls_label_file = os.path.join(gt_path, 'image_class_labels.txt')
    gt_file = os.path.join(gt_path, 'bounding_boxes.txt')
    gts = pd.read_csv(gt_file, header=None, sep=" ", names=["id",
                                                            "x","y","w","h"])
    labels = pd.read_csv(img_cls_label_file, header=None, sep=" ", names=["id",
                                                                          "label"])

    cur_gt = gts[gts['id'] == im_name]
    cur_label = labels[labels['id'] == im_name]['label']
    x1 = cur_gt['x']
    y1 = cur_gt['y']
    x2 = cur_gt['w'] + x1
    y2 = cur_gt['h'] + y1
    # pdb.set_trace()
    gt_bbxs = np.asarray([x1,y1,x2,y2]).reshape(1,-1)
    return gt_bbxs, labels

def extract_gt_Imagenet(targets, im_name):
    gt_path = "/vulcanscratch/rssaketh/LOST/"
    box_file = os.path.join(gt_path, 'imagenet_box_annotations.pkl')
    boxes = pickle.load(open(box_file,'rb'))
    cbox = boxes[im_name]
    labels = os.path.dirname(im_name)
    # pdb.set_trace()
    gt_bbxs = np.asarray(cbox)
    return gt_bbxs, labels

def extract_gt_DUT_OMRON(targets, im_name):
    gt_path = "saliency/data/DUT-OMRON/pixelwiseGT-new-PNG/"
    im_name = im_name.split('.')[0] + '.png'
    if not os.path.isfile(os.path.join(gt_path, im_name)):
        raise Exception("Gt file {} not found".format(im_name))
    img = np.asarray(Image.open(os.path.join(gt_path, im_name)).convert('L'))
    if np.unique(img).shape[0] > 2:
        img[img > 0] = 255
    return img, None

def extract_gt_LVIS(targets):
    objects = targets
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        # Remove iscrowd boxes
        gt_cls = objects[o]["category_id"]
        gt_clss.append(gt_cls)
        bbx = objects[o]["bbox"]
        x1y1x2y2 = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        x1y1x2y2 = [int(round(x)) for x in x1y1x2y2]
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss


def extract_gt_COCO(targets, remove_iscrowd=True):
    objects = targets
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        # Remove iscrowd boxes
        if remove_iscrowd and objects[o]["iscrowd"] == 1:
            continue
        gt_cls = objects[o]["category_id"]
        gt_clss.append(gt_cls)
        bbx = objects[o]["bbox"]
        x1y1x2y2 = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        x1y1x2y2 = [int(round(x)) for x in x1y1x2y2]
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss


def extract_gt_VOC(targets, remove_hards=False):
    objects = targets["annotation"]["object"]
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        if remove_hards and (
            objects[o]["truncated"] == "1" or objects[o]["difficult"] == "1"
        ):
            continue
        gt_cls = objects[o]["name"]
        gt_clss.append(gt_cls)
        obj = objects[o]["bndbox"]
        x1y1x2y2 = [
            int(obj["xmin"]),
            int(obj["ymin"]),
            int(obj["xmax"]),
            int(obj["ymax"]),
        ]
        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        x1y1x2y2[0] -= 1
        x1y1x2y2[1] -= 1
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss


def bbox_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (list or tuple): Bounding box in format [x, y, w, h].
        boxB (list or tuple): Bounding box in format [x, y, w, h].

    Returns:
        float: The IoU value, a number between 0 and 1.
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2] format
    # (x, y) is the top-left corner, (x2, y2) is the bottom-right corner
    boxA_x1 = boxA[0]
    boxA_y1 = boxA[1]
    boxA_x2 = boxA[0] + boxA[2] # x + width
    boxA_y2 = boxA[1] + boxA[3] # y + height

    boxB_x1 = boxB[0]
    boxB_y1 = boxB[1]
    boxB_x2 = boxB[0] + boxB[2] # x + width
    boxB_y2 = boxB[1] + boxB[3] # y + height

    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(boxA_x1, boxB_x1)
    inter_y1 = max(boxA_y1, boxB_y1)
    inter_x2 = min(boxA_x2, boxB_x2)
    inter_y2 = min(boxA_y2, boxB_y2)

    # Compute the area of intersection rectangle
    # If there is no overlap, the width or height will be negative, max(0, ...) handles this
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth bounding boxes
    boxA_area = boxA[2] * boxA[3] # width * height
    boxB_area = boxB[2] * boxB[3] # width * height

    # Compute the area of union by subtracting the intersection area
    # from the sum of the areas
    union_area = boxA_area + boxB_area - inter_area

    # Handle cases where union area is zero (e.g., both boxes are zero-area)
    if union_area == 0:
        return 0.0

    # Compute the IoU
    iou = inter_area / union_area

    return iou


def bbox_iou__(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):

    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def select_coco_20k(sel_file, all_annotations_file):
    print('Building COCO 20k dataset.')

    # load all annotations
    with open(all_annotations_file, "r") as f:
        train2014 = json.load(f)

    # load selected images
    with open(sel_file, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]
    im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]

    new_anno = []
    new_images = []

    for i in tqdm(im20k):
        new_anno.extend(
            [a for a in train2014["annotations"] if a["image_id"] == int(i)]
        )
        new_images.extend([a for a in train2014["images"] if a["id"] == int(i)])

    train2014_20k = {}
    train2014_20k["images"] = new_images
    train2014_20k["annotations"] = new_anno
    train2014_20k["categories"] = train2014["categories"]

    with open("datasets/instances_train2014_sel20k.json", "w") as outfile:
        json.dump(train2014_20k, outfile)

    print('Done.')

