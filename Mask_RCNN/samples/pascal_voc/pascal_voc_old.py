"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Training on VOC Written by genausz(genausz@hotmail.com)
-----------------------------------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a model from coco weights.
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=coco --year=2012
    # Train a new model starting from ImageNet weights.
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=imagenet --year=2012
    # Continue training a model that you had trained earlier
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=/path/to/weights.h5  --year=2012
    # Continue training the last model you trained
    python3 voc.py train --dataset=/path/to/VOCdevkit/ --model=last
    # Run VOC inference on the last model you trained
    python3 voc.py inference --dataset=/path/to/VOCdevkit/ --model=last --year=2012 --limit=50
"""



import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from bs4 import BeautifulSoup as bs
import cv2
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Inference result directory
RESULTS_DIR = os.path.abspath("./inference/")  # 後面程式碼會自己產生這個資料夾

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

import matplotlib
# Agg backend runs without a display
matplotlib.use('Agg')  # 不知道是什麼
import matplotlib.pyplot as plt

import json
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
DEFAULT_DATASET_YEAR = '2012'  # 我可能不需要這個

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")  
# 這邊之後要改，不能用coco pretrain的model


# VOC DATASET MASK MAP FUNCTION
# Following codes are mapping each mask color(SegmentationClass) to ground truth index.
# - reference: https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] #potted plant # tv/monitor

def build_colormap2label():
    """Build a RGB color to label mapping for segmentation."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map a RGB color to a label."""
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
# VOC DATASET MASK MAP FUNCTION


class VocConfig(Config):
    NAME = "voc"

    IMAGE_PER_GPU = 2

    NUM_CLASSES = 1 + 20 # VOC 2012 have 20 classes. "1" is for background.

class InferenceConfig(VocConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 2
    IMAGES_PER_GPU = 32  # 8 # 1
    DETECTION_MIN_CONFIDENCE = 0  # 這邊感覺可以調一下


class VocDataset(utils.Dataset):
    def load_voc(self, dataset_dir, trainval, year='2012'):  # year 2012應該可以拿掉
        """Load a voc_year of the VOC dataset.
        dataset_dir: The root directory of the VOC dataset, example: '/mnt/disk1/VOCdevkit'
        trainval: 'train' or 'val' for Training or Validation
        year: '2007' or '2012' for VOC dataset
        """
                
        PATH = os.path.join(ROOT_DIR, "dataset/pascal_train.json")
        voc_json = COCO(PATH) # load training annotations
        image_ids = []
        if trainval == 'train':
            image_ids = list(voc_json.imgs.keys())[:1080]
        else:
            image_ids = list(voc_json.imgs.keys())[1080:]
        for image_id in image_ids:
            image_file_name = voc_json.imgs[736]['file_name']
            image_dir_path = os.path.join(ROOT_DIR, "dataset/train_images/")
            image_path = os.path.join(image_dir_path, image_file_name)
            self.add_image("voc",
                            image_id=image_file_name,
                            path=image_path)

    def load_class_label(self, image_id):
        '''Mapping SegmentationClass image's color to indice of ground truth 
        image_id: id of mask
        Return:
        class_label: [height, width] matrix contains values form 0 to 20
        '''
        PATH = os.path.join(ROOT_DIR, "dataset/pascal_train.json")
        voc_json = COCO(PATH) # load training annotations
        height = voc_json.imgs[image_id]['height']
        width = voc_json.imgs[image_id]['width']
        
        class_label = np.zeros(height, width)
        annids = voc_json.getAnnIds(imgIds=image_id)
        anns = voc_json.loadAnns(annids)
        for i in range(len(annids)):
            mask = voc_json.annToMask(anns[i])
            cate = anns[i]['category_id']
            mask = mask * cate
            class_label = class_label + mask
        print('class_label = ', class_label.shape)
        return class_label

    def load_mask(self, image_id):
        '''Mapping annotation images to real Masks(MRCNN needed)
        image_id: id of mask
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        '''
        PATH = os.path.join(ROOT_DIR, "dataset/pascal_train.json")
        voc_json = COCO(PATH) # load training annotations
        
        annids = voc_json.getAnnIds(imgIds=image_id)
        classes_ids = np.array(annids)
        anns = voc_json.loadAnns(annids)
        masks = []
        for i in range(len(annids)):
            mask = voc_json.annToMask(anns[i])
            mask = mask >= 1
            masks.append(mask)
        masks = np.array(masks).transpose((1,2,0))
        print('masks = ', masks.shape)
        print('classes_ids = ', classes_ids.shape)
        return masks, classes_ids


############################################################
#  Inference
############################################################

def inference(model, dataset, limit):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    time_dir = "{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    time_dir = os.path.join(RESULTS_DIR, time_dir)
    os.makedirs(time_dir)

    # Load over images
    for image_id in dataset.image_ids[:limit]:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        # Save image with masks
        if len(r['class_ids']) > 0:
            print('[*] {}th image has {} instance(s).'.format(image_id, len(r['class_ids'])))
            visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                dataset.class_names, r['scores'],
                show_bbox=True, show_mask=True,
                title="Predictions")
            plt.savefig("{}/{}".format(time_dir, dataset.image_info[image_id]["id"]))
            plt.close()
        else:
            plt.imshow(image)
            plt.savefig("{}/noinstance_{}".format(time_dir, dataset.image_info[image_id]["id"]))
            print('[*] {}th image have no instance.'.format(image_id))
            plt.close()



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on PASCAL VOC.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference' on PASCAL VOC")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/voc/",
                        help='Directory of the PASCAL VOC dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the PASCAL VOC dataset (2007 or 2012) (default=2012)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'voc'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/", 
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=10,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=10)')

    # TODO
    '''
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip PASCAL VOC files (default=False)',
                        type=bool)
    '''
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    #print("Auto Download: ", args.download)


    # Configurations
    if args.command == "train":
        config = VocConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)


    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_WEIGHTS_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)


    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = VocDataset()
        dataset_train.load_voc(args.dataset, "train", year=args.year)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VocDataset()
        dataset_val.load_voc(args.dataset, "val", year=args.year)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "inference":
        #print("evaluate have not been implemented")
        # Validation dataset
        dataset_val = VocDataset()
        voc = dataset_val.load_voc(args.dataset, "val", year=args.year)
        dataset_val.prepare()
        print("Running voc inference on {} images.".format(args.limit))
        inference(model, dataset_val, int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))