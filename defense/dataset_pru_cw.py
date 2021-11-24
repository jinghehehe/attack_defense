import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import glob
import logging
import os
from PIL import Image


def load_data(args, item=None):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Resize([320,320])])

    #img_dir = '/data/private_data/jf_private/CCPD2020/CCPD2019/defense/'
    img_dir = '/data/private_data/jf_private/CCPD2020/CCPD2019/defense/resnet_16/'
    #img_dir = '/home/jf/detect/CCPD/images/defense_adv/'
    train_dataset = CCPDdataset(img_dir, sub_item=item, debug=False, transform=transform)

    val_dataset = CCPDdataset(img_dir, sub_item=item, debug=True, train=False, transform=transform, pru=0)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=int(args.workers),
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=int(args.workers),
                                             drop_last=False)

    return train_loader, val_loader


class CCPDdataset(Dataset):
    def __init__(self, root=None, transform=None, train=True, sub_item=['res'],
                 rng_seed=0, pru=False, debug=False):
        self.img_dir = root
        print('image dir:', self.img_dir)
        self.img_paths = []
        self.train = train
        self.sub_item = sub_item
        self.debug = debug
        self.transform = transform
        self.pru = 0
        self.rng = np.random.RandomState(rng_seed)
        
        if self.train:
            self.adv_path = self.img_dir + 'adv'
            self.ori_path = self.img_dir + 'ori'
            
            ##self.adv_path = self.img_dir
            # json_list = glob.glob(self.img_dir+ sub_item[0] + '_train.json')
        else:
            ##self.adv_path = self.img_dir
           
            self.adv_path = self.img_dir + 'adv'
            self.ori_path = self.img_dir + 'ori'
            # print('validation!')
            # json_list = glob.glob(self.img_dir+ sub_item[0] + '_val.json')
            # print(json_list)
        
        metas = []
        for filename in os.listdir(self.adv_path):
            ##name = self.adv_path+filename
            name = self.adv_path+"/"+filename
            #print(name)
            metas.append(name)
        
        #print(metas)
        # metas2 = []
        # for filename in os.listdir(self.ori_path):
        #     metas2+= (self.ori_path+'/'+filename)   
        # for sub_json in json_list:
        #     if 1==1 or sub_json.split('_')[-2] in sub_item:
        #         with open(sub_json, 'r') as file:
        #             metas += json.load(file) 
        #             logging.info("load data {} {}".format(sub_json,len(metas)))
                    
        self.jdata = metas
        self.rng.shuffle(self.jdata)
        del metas

        logging.info('--sub-{}----{}---\n'.format(self.sub_item,len(self.jdata)))
    
    def __len__(self):
        return len(self.jdata)
    
    def __getitem__(self, index):
        """
        
        :param index:
        :return: adversarial image & gt image
        """
        img_name = self.jdata[index]
        print(img_name)
        gt_name = img_name.replace('adv', 'ori')
        print(gt_name)
        """对抗图片"""
        img = cv2.imread(img_name)
        """原始图片"""
        gt = cv2.imread(gt_name)
        if gt is None:
            gt = img

        if self.transform is not None:
            img = img[:, :, ::-1]
            img = Image.fromarray(img)
            img = self.transform(img)

            gt = gt[:, :, ::-1]
            gt = Image.fromarray(gt)
            gt = self.transform(gt)

        return img, gt