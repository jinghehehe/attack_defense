#!/bin/bash 

CUDA_VISIBLE_DEVICES=1 python train.py --img 640 --batch 16 --epochs 5 --data license_plate.yaml --weights yolov3-spp.pt

#CUDA_VISIBLE_DEVICES=1 python train.py --img 640 --batch 16 --epochs 20 --data license_plate.yaml --cfg *.yaml --weights 'kk'