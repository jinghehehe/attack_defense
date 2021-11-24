import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from utils import (weights_init, save_checkpoint, load_checkpoint, AverageMeter)
from config import cfg
import logging
import time
from unet import UNet


class UDenoiser:
    def __init__(self, args, writer, device, image_nc, save_path, batch_per_epoch, train=True,
                 LPR=None):
        self.device = device
        self.input_nc = image_nc
        self.box_min = -1
        self.box_max = 1
        self.save_path = save_path
        self.writer = writer
        self.batch_per_epoch = batch_per_epoch
        self.epoch = 0
        self.training = train
        self.lr = args.lr if hasattr(args, 'lr') else 1e-4
        self.args = args
        self.net = UNet(bn=args.bn_type).to(device)
        self.LPR = LPR
        self.LPR.eval()

        logging.info("net:{}".format(self.net))
        self.y_one = torch.ones(args.batch_size, device=self.device)
        self.y_zero = torch.zeros(args.batch_size, device=self.device)

        if self.training:
            # initialize optimizers
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=args.lr,betas=(args.betas1, 0.999))
            prams = sum([p.nelement() for p in self.net.parameters()])
            logging.info("all_prams-- net {}".format(prams/1e6))

        self.criterion_ce = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_bce = nn.BCEWithLogitsLoss().cuda()
        self.criterion_mse = nn.MSELoss().cuda()

        if args.resume is '':
            logging.info('Initializing weights...')
            self.net.apply(weights_init)
        else:
            self.load_networks(load_root=args.resume)

    def backward(self):
        self.optimizer.zero_grad()
        if self.epoch in cfg.step_lam_adv:
            cfg.lam_adv = cfg.step_lam_adv[self.epoch]

        self.loss_l1 = cfg.lam_l1 * F.smooth_l1_loss(self.rc_images, self.gt_images)
        if self.LPR:     # 特征层的重构损失
            self.loss_features = cfg.lam_l1 * F.smooth_l1_loss(self.rc_features, self.gt_features)
        else:
            self.loss_features = 0

        self.loss = self.loss_l1 + 1.0 * self.loss_features
        self.loss.backward()
        self.optimizer.step()

    def set_input(self, image, labels):
        self.input = image
        self.gt_images = labels
        rc_images = self.net(self.input, cat_img=False)
        self.rc_images = torch.clamp(rc_images, self.box_min, self.box_max)
        if self.LPR:
            self.rc_features = self.LPR(self.rc_images, True)
            self.gt_features = self.LPR(self.gt_images, True)

    def set_lr(self, optimizers, epoch):
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if epoch in [10, 20]:
            self.lr = self.lr*self.args.gamma
            for optim in optimizers:
                for param_group in optim.param_groups:
                    param_group['lr'] = self.lr
        return self.lr

    def train_batch_(self,images,labels,iteration):
        self.set_input(images,labels)
        if iteration % self.args.gratio == 0:
            self.backward()

    def train(self, train_dataloader, epochs):
        for epoch in range(self.epoch, epochs):
            loss_l1_sum = AverageMeter()
            loss_F_sum = AverageMeter()
            loss_all = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            self.epoch = epoch

            lr = self.set_lr([self.optimizer], epoch)
            s_time = time.time()
            for iteration, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                data_time.update(time.time()-s_time)
                self.train_batch_(images, labels, iteration)

                loss_l1_sum.update(self.loss_l1.item())
                loss_F_sum.update(self.loss_features.item())
                loss_all.update(self.loss.item())

                batch_time.update(time.time()-s_time)
                s_time = time.time()

                if iteration % cfg.print_freq ==0 and iteration!=0:
                    logging.info('iter: [{}/{}/{}]  Time: {:.2f}/{:.2f}\t'        
                        'L1 ({loss_l1.avg:.4f})\t' 
                        'LFeatures ({loss_F.avg:.4f})\t' 
                        'L_ALL ({loss_all.avg:.4f})\t'.format(
                        epoch, iteration, self.batch_per_epoch, batch_time.avg, data_time.avg,
                        loss_l1=loss_l1_sum, loss_F=loss_F_sum, loss_all=loss_all)
                    )

            self.writer.add_scalar('loss/l1', loss_l1_sum.avg, epoch)
            self.writer.add_scalar('loss/lF', loss_F_sum.avg, epoch)
            self.writer.add_scalar('loss/loss', loss_all.avg, epoch)
            self.writer.add_scalar('lr/lr', lr, epoch)
            if (epoch != 0) and epoch % cfg.saveInterval == 0:
                self.save_networks(epoch)

    def save_networks(self, epoch, **kwargs):
        save_checkpoint(self.net, self.optimizer, save_dir=self.save_path,
                        name='{}_net.pth'.format(epoch))

    def load_networks(self, load_root, **kwargs):
        if self.training:
            epoch = load_root.split('/')[-1].split('_')[0]
            load_root = os.path.dirname(load_root)
            self.epoch = int(epoch)
            load_path = os.path.join(load_root,'{}_net.pth'.format(epoch))
            load_checkpoint(model=self.net,path=load_path,optimizer=self.optimizer)

        else:
            load_checkpoint(model=self.net,path=load_root)
