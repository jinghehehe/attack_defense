import torch
from denoiser import UDenoiser
import dataset_pru_cw
from utils2 import setup_logging
import argparse, os
from datetime import datetime
import logging
from model import VGG, ResNet, DenseNet121, BotNet50

parser = argparse.ArgumentParser()

# data params
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--prudata', type=float, default=0.0, help='number of epochs to train for')
parser.add_argument('--steps', type=int, default=50, help='the learning_rate decay steps')

# model params
parser.add_argument('--model', type=str, default='unet', help='unet')
parser.add_argument('--weight', type=str, default='/home/jf/detect/yolov3/defense/LPR_weights/resnet_145.pth')
parser.add_argument('--bn_type', type=str, default='bn')
parser.add_argument('--resume', type=str, default='')

# optim params
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--optim', type=str, default='rmsp')
parser.add_argument('--gamma', type=float, default=0.2, help='the learning_rate deacy gamma')
parser.add_argument('--gratio', type=int, default=1)
parser.add_argument('--betas1', type=float, default=0.5)
parser.add_argument('--wgan_clamp', type=float, default=0.01)

# log param
parser.add_argument('--save', type=str, default='', help='Where to store samples and models')
parser.add_argument('--results_dir', default='/data/private_data/jf_private/CCPD2020/CCPD2019/defense/results', help='Where to store samples and models')
parser.add_argument('--gpus', default='3', type=str, help='use gpus idx, 0,1,2,3')
parser.add_argument('--sub_data', default='green, yellow, base,blur,challenge,db,fn,rotate,tilt,weather', type=str)

args = parser.parse_args()
args.gpus = [int(x) for x in args.gpus.split(',')]
#os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpus[0])
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(device,lpr=None,lpr2=None,lpr3=None):
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        print('mkdir:', save_path)
        os.makedirs(save_path)
    
    # setup_logging(os.path.join(save_path, 'log.txt'))
    # writer = SummaryWriter(log_dir=save_path)

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)
    #logging.info("run cfg: {}".format(cfg.__dict__))

    item = ['res']
    train_dataloader, val_dataloader = dataset_pru_cw.load_data(args, None)
    
    batch_per_epoch = len(train_dataloader)
    # BUILD & LOAD LPR MODEL
    # if 'dark' in args.weight:
    #     lpr = Yolov3()
    # elif 'res' in args.weight:
    #     lpr = ResNet()
    # elif 'vgg' in args.weight:
    #     lpr = VGG()
    # elif 'dense' in args.weight:
    #     lpr = DenseNet121()
    # else:
    #     lpr = BotNet50()

    # lpr.load_state_dict(torch.load(args.weight, map_location='cpu')['state_dict'])
    lpr = lpr.cuda().eval()

    de = UDenoiser(args, device,3, save_path, batch_per_epoch, LPR=lpr,LPR2=lpr2,LPR3=lpr3)
    de.train(train_dataloader, args.epochs)


if __name__ == '__main__':
    init_model()
