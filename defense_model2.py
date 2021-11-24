import argparse
import json
import os
from pathlib import Path
from threading import Thread
import torch.nn as nn
import numpy as np
import torch
import yaml
from tqdm import tqdm
import sys

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.loss import compute_loss,compute_loss2
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
### 引入攻击部分
sys.path.append('./defense/')
from unet import UNet
from train2 import init_model
# sys.path.append('../')
# from attack.tog.attacks import TOG
from PIL import Image

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model0=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         log_imgs=0):  # number of logged images

    # Initialize/load model and set device
    training = model0 is not None
    if training:  # called by train.py
        device = next(model0.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        
        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model0 = attempt_load(weights, map_location=device)  # load FP32 model
        print(model0)
        model = nn.Sequential(*list(model0.children())[0][:11])
        model2 = nn.Sequential(*list(model0.children())[0][:9])
        model3 = nn.Sequential(*list(model0.children())[0][:7])
        #print(model0)
        imgsz = check_img_size(imgsz, s=32)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    model2.eval()
    model3.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

#     # Dataloader
#     if not training:
#         img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#         #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
#         path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
#         dataloader = create_dataloader(path, imgsz, batch_size, 32, opt,  pad=0.5, rect=True,
#                                        prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]
# # dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True,
# #                                        prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]\
# # 改了3处 97,99m106
#     seen = 0
#     confusion_matrix = ConfusionMatrix(nc=nc)
#     #names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
#     coco91class = coco80_to_coco91_class()
#     s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
#     p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
#     loss = torch.zeros(3, device=device)
#     jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
#     for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
#         np.set_printoptions(threshold=np.inf)
#         img = img.to(device, non_blocking=True)
#         img = img.half() if half else img.float()  # uint8 to fp16/32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         #print(targets)
#         # ## 引入攻击部分
#         # attack_model = TOG(model)
#         # #targets2 = targets.clone()

#         # targets2 = torch.empty(0,6)
#         # # print(111)
#         # # print(targets2)
#         # # print(111)
#         # img = attack_model.tog_vanishing(img, targets2, compute_loss2)
#         # img = img.to(device, non_blocking=True)
#         # img = img.half() if half else img.float()  # uint8 to fp16/32

#         # ## 引入攻击部分
#         #print(targets)
#         # ## 引入防御部分
    init_model(device,model.float(),model2.float(),model3.float())
        # d_block = UNet()
        # d_block.load_state_dict(torch.load('/data/private_data/jf_private/CCPD2020/CCPD2019/defense/results/2021-06-03_16-06-10/2_net.pth', map_location='cpu')['model'])
        # d_block.cuda().eval()
        # img = d_block(img)
        #defense_model = TOG(model)
        #targets2 = targets.clone()

        #targets2 = torch.empty(0,6)

        # img = defense_model.tog_vanishing(img, targets, compute_loss2)
        # img = img.to(device, non_blocking=True)
        # img = img.half() if half else img.float()  # uint8 to fp16/32

        # ## 引入防御部分
        #init_model(device)


if __name__ == '__main__':
    parser2 = argparse.ArgumentParser(prog='test.py')
    parser2.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp41/weights/best.pt', help='model.pt path(s)')
    parser2.add_argument('--data', type=str, default='./data/license_plate.yaml', help='*.data path')
    parser2.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser2.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser2.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser2.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser2.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser2.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser2.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser2.add_argument('--augment', action='store_true', help='augmented inference')
    parser2.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser2.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser2.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser2.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser2.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser2.add_argument('--project', default='runs/test', help='save to project/name')
    parser2.add_argument('--name', default='exp', help='save to project/name')
    parser2.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser2.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements()

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(f, x)  # plot
