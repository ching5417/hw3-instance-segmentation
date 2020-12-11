# hw3-instance-segmentation
Code for NCTU Selected Topics in Visual Recognition using Deep Learning Homework 3

## Hardware
The following specs were used to create the original solution.
- Ubuntu 20.04 LTS
- Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz (8 Cores)
- 2x NVIDIA GeForce GTX 1080

## Reproducing Submission
To reproduct the training and testing process, do the following steps:
1. [Installation](#installation)
2. [Training](#training)
3. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt.
```
pip install -r requirements.txt
```

## Dataset Preparation
All required files are already in data directory.

## Training

### Train models
To train models, run following commands.

```
$ cd Mask_RCNN/samples/pascal_voc/
$ python3 pascal_voc.py \
  train \
  --model=imagenet
```
ex: `python3 pascal_voc.py train --model=imagenet`

After training, the model will save to `Mask_RCNN/logs/` folder.
```
model
+- mask_rcnn_coco_0110.h5
```
[Here](https://drive.google.com/drive/folders/17wW7i1jNhWoWXyMpot97sT2rHYj3tM6g?usp=sharing) download the model (ex: `mask_rcnn_coco_0110.h5`)

## Make Submission
Following command will ensemble of all models and make submissions.
python3 pascal_voc.py evaluate
```
$ python3 pascal_voc.py \
  evaluate\
  --model=model.h5
```
ex: `python3 pascal_voc.py evaluate --model=model/mask_rcnn_coco_0110.h5`

After testing, it will output a `submission.json` in `Mask_RCNN/output/` folder.

I also put my testing result [here](https://drive.google.com/drive/folders/1SdG-fKzXxGuGdml9I5aXNkObh0DPv_a0?usp=sharing).
