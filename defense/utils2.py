
#!/usr/bin/python
# encoding: utf-8

import torch.nn.init as init
import torch.nn as nn
import torch

import os,sys
import cv2
import numpy as np

from bisect import bisect_right
import collections
import logging
from collections import OrderedDict
import math
from PIL import Image, ImageDraw, ImageFont
import shutil


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    if os.path.exists(log_file):
        new_log = log_file.rsplit('.',1)[0] + '_bak.txt'
        shutil.move(log_file,new_log)
        print('{} is exit,move to {} \n'.format(log_file,new_log))
    
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m,'bias') and m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def save_checkpoint(model,optimizer=None,scheduler=None,save_dir = '.',name = 'checkpoint.pth',**kwargs):
    import os
    data = {}
    data["model"] = model.state_dict()
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    data.update(kwargs)
    
    save_file = os.path.join(save_dir, name)

    logging.info("Saving checkpoint to {}".format(save_file))
    torch.save(data, save_file)

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def load_checkpoint(model,path=None,optimizer=None,scheduler=None,trans=None,**kwargs):
    assert path!=None,'pre_train model path can\'t be None'
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    
    model_state = strip_prefix_if_present(checkpoint['model'],prefix="module.")
    
    if trans is not None:
        new_keys = list(model.state_dict().keys())
        stripped_state_dict = OrderedDict()
        i = 0
        for key, value in model_state.items():
            stripped_state_dict[new_keys[i]] = value
            i+=1
        model.load_state_dict(stripped_state_dict)
    else:
        model.load_state_dict(model_state)

    if "optimizer" in checkpoint and optimizer!=None:
        logging.info("Loading optimizer from {}".format(path))
        optimizer.load_state_dict(checkpoint.pop("optimizer"))
    if "scheduler" in checkpoint and scheduler!=None:
        logging.info("Loading scheduler from {}".format(path))
        scheduler.load_state_dict(checkpoint.pop("scheduler"))
    logging.info("Loading model from {}".format(path))
    
    return checkpoint

def torch_summarize(model, show_weights=True, show_parameters=True):
    from torch.nn.modules.module import _addindent

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
            # print(modstr)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)
        
        params = sum([p.nelement() for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params/1e6)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr



def pad_resize(image,target_size):
        img_h,img_w,_ = image.shape
        if img_h!=img_w:
            print(img_h,img_w)
            max_len = max(img_w,img_h)
            min_len = min(img_h,img_w)
            pad = (max_len - min_len)//2
            ret = np.zeros((max_len,max_len,3), dtype=image.dtype)
            # ret[:,:,:] = [0,255,0]
            print(ret.shape)
            if img_h == max_len:
                ret[:,pad:pad+img_w,:] = image
            elif img_w == max_len:
                ret[pad:pad+img_h,:,:] = image
        else:
            ret = image
        ret = cv2.resize(ret,(target_size,target_size))
        return ret

def put_text(img,text,loc,color=(0,255,0),size=20,font_path = None):
    if font_path is None:
        foot_path = '/home/cy/wjq_project/glp/font/platech.ttf'
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    font = ImageFont.truetype(foot_path, size)
    draw = ImageDraw.Draw(pil_img)
    
    draw.text(loc, text, font = font, fill = color)
    img = cv2.cvtColor(np.asarray(pil_img),cv2.COLOR_RGB2BGR)
    return img


class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
 
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts



def montage(imgs, border_size):
    assert len(imgs) > 0
    shapes = [i.shape for i in imgs]
    assert all(s == shapes[0] for s in shapes), shapes
    n = len(imgs)
    nh = nw = int(math.ceil(math.sqrt(n)))
    ishape = imgs[0].shape
    vert_border_shape = (ishape[0], border_size) + ishape[2:]
    vert_border_bar = (np.ones(vert_border_shape) * 128).astype('uint8')

    hori_border_shape = (border_size, ishape[1] * nw + border_size * (nw - 1)) + ishape[2:]
    hori_border_bar = (np.ones(hori_border_shape) * 128).astype('uint8')

    rows = []
    for i in range(0, n, nw):
        row = imgs[i:i + nw]
        stacks = []
        for j in range(len(row)):
            img = row[j]
            if j > 0:
                stacks.append(vert_border_bar)
            stacks.append(img)
        while len(stacks) < nw * 2 - 1:
            stacks.append(vert_border_bar)
            stacks.append(np.zeros(img.shape, 'uint8'))
        row_img = np.hstack(stacks)

        if i > 0:
            rows.append(hori_border_bar)
        rows.append(row_img)

    return np.vstack(rows)
