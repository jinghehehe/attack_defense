#!/bin/bash 

CUDA_VISIBLE_DEVICES=1 python train.py --img 640 --batch 8 --epochs 20 --data license_plate.yaml --cfg 'vgg.yaml' --weights 'kk'