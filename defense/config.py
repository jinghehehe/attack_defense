import os,sys
import torch
import torchvision.transforms as transforms

index2chrs = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
              "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
              "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "使", 
              "澳", "港", "警", "挂", "学", "领", 
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
              "A","B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
              "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z","#"]

chars = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "使": 31, "澳": 32, "港": 33, "警": 34, "挂": 35, "学": 36,
         "领": 37, "0": 38, "1": 39, "2": 40, "3": 41, "4": 42, "5": 43,
         "6": 44, "7": 45, "8": 46, "9": 47, "A": 48, "B": 49, "C": 50, "D": 51, "E": 52, "F": 53, "G": 54, "H": 55,
         "J": 56, "K": 57, "L": 58, "M": 59, "N": 60, "P": 61, "Q": 62, "R": 63, "S": 64, "T": 65, "U": 66, "V": 67,
         "W": 68, "X": 69, "Y": 70, "Z": 71}

alphabet = ''.join(index2chrs)

nclass = len(alphabet)

def to_pln(label):
    return ''.join([alphabet[x] for x in label])

nc = 3
nh = 256
img_h = 32
lr = 1e-3
optimizer = 'SGD'

saveInterval = 2
print_freq = 10


class Config(object):
    def __init__(self):
        
        self.nclass = len(index2chrs)
        self.img_nc = 3
        self.nh = 256
        self.img_h = 32
        
        # train config
        self.BOX_MIN = -1
        self.BOX_MAX = 1
        self.saveInterval = 1
        self.print_freq = 10

        # loss and optim param
        
        self.loss_items = ['l1']
        
        # self.adv_obj = [3,6]
        # self.target = 45
        
        self.data_nums = 150000
        self.gnr_nums = 0
        
        self.lam_l1 = 1
        self.lam_ssim = 0.01
        # self.lam_per = 0.01
        self.lam_per = 1   # TODO 0.5 / 1

        # self.step_lam_adv = {20:10,50:30,100:50,150:100}
        # self.step_lam_adv = {20:5,50:10,100:50,150:100}
        
        # self.step_lam_adv = {20:5,50:10,100:50,150:200}
        
        #TODO wgan
        self.step_lam_adv = {20:1,50:2,100:3,150:3}
        self.lam_adv = 1
        
        self.lam_gan = 1
        self.max_thre = 0.0
        self.wgan_gp = 10

        self.per_bound = 0.01
        self.patch_dir = ''
        
        # root = '/data/private_data/wjq_private/2_CRNN/wjq_cnn_ce_ccpd_8ce_v2/'
        # self.target_model_path= root + 'results/ccpd_cnn_b256_8ce_rgb_22w_pru_data/crnn_plate_50.pth'
        
        root = '/data/private_data/wjq_private/2_CRNN/wjq_cnn_ce_ccpd_8ce_v3/'
        self.target_model_path= root + 'results/ccpd_22w_cnnpool_gnr7w_8ce_73/crnn_plate_55.pth'
        
        self.img_dir = '/data/private_data/wjq_private/CCPD2019/ccpd_rec_all_8/ccpd_split_res/'


cfg = Config()

if __name__ == '__main__':
    print(1111)
    dict_ = {}
    for name in cfg.__dict__:
        if not name in ['provinces','alphabets','alphabet','ads']:
            dict_[name] = getattr(cfg,name)
    print(dict_)