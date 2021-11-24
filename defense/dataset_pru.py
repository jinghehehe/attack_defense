import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import json, glob
import logging


def load_data(args):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_dir = '/data/private_data/jf_private/CCPD2019/ccpd_rec_all_8/ccpd_split_res/'
    item = ['yellow', 'green', 'base', 'blur', 'challenge', 'db', 'fn', 'rotate', 'tilt', 'weather']

    train_dataset = CCPDdataset(img_dir, sub_item=item, debug=False, transform=transform)
    val_dataset = CCPDdataset(img_dir, sub_item=item, debug=True, train=False, transform=transform, pru=0)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=int(args.workers), drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=int(args.workers),drop_last=True)

    return train_loader, val_loader


class CCPDdataset(Dataset):
    def __init__(self, root=None, transform=None, train=True,
                 sub_item=None, rng_seed=0, pru=False, debug=False):
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
            json_list = glob.glob(self.img_dir+sub_item[0]+'_train.json')
        else:
            print('Validation Time!')
            json_list = glob.glob(self.img_dir+sub_item[0]+'_val.json')
            print(json_list)
        
        metas = []
        for sub_json in json_list:
            if 1 == 1 or sub_json.split('_')[-2] in sub_item:
                with open(sub_json, 'r') as file:
                    metas += json.load(file) 
                    logging.info("load data {} {}".format(sub_json, len(metas)))
                    
        self.jdata = metas
        self.rng.shuffle(self.jdata)
        del metas

        print('Number of images:', len(self.jdata))
        logging.info('--sub-{}----{}---\n'.format(self.sub_item, len(self.jdata)))
    
    def __len__(self):
        return len(self.jdata)
    
    def __getitem__(self, index):
        img_name = self.jdata[index]['img_name']
        img = cv2.imread(img_name)
        iname = img_name.rsplit('/', 1)[-1].split('.')[0]
        label = [int(x) for x in iname.split('_')[:-1]]
        label = torch.LongTensor(label)
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label