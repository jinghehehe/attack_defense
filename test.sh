#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python test.py --img 640 --batch 16  --data ./data/license_plate.yaml --weights ./runs/train/exp3/weights/best.pt --task val

# --weights ./runs/train/exp3/weights/best.pt yolov3
# --weights ./runs/train/vgg163/weights/best.pt vgg16
# --weights ./runs/train/mobile2/weights/best.pt mobile
# --weights ./runs/train/densenet5/weights/best.pt densenet
# --weights ./runs/train/resnet/weights/best.pt vgg16
# --weights ./runs/train/transformer/weights/best.pt transformer