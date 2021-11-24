from utils import *
import torchvision.transforms as transforms
import time
import config
from model import VGG, ResNet, DenseNet121, BotNet50
import argparse
import os
from dataset_pru import CCPDdataset
from unet import UNet

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--defense', type=int, help='whether to use defense block', default=0)
parser.add_argument('--defense_weight', type=str, default='defense_weights/res_18.pth', help='path of defense weight')
parser.add_argument('--weight', type=str, default='LPR_weights/dense_135.pth', help='path of lpr weight')
parser.add_argument('--gpus', default='2', type=str, help='use gpus idx, 0,1,2,3')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpus)
print("using gpus {}".format(args.gpus))


def val(model, dataloader, d_block):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    defense_time = 0.01
    model.eval()  # 模型要固定参数！
    correct = 0
    for i, data in enumerate(dataloader):
        n_correct = 0
        inputs, target = data
        input_var = inputs.cuda()
        data_time.update(time.time() - end)
        # FORWARD
        with torch.no_grad():
            if d_block is not None:    # DEFENSE_BLOCK, TEST ACC AFTER DEFENSE
                t0 = time.time()
                input_var = d_block(input_var)
                t1 = time.time()
                defense_time += t1-t0   # TEST DEFENSE_BLOCK SPEED, 100 FPS
                #print('D-TIME:', t1-t0)
            t2 = time.time()
            preds = model(input_var)
            t3 = time.time()
            #print('LPR-TIME:', t3-t2)
        #print((i+1)/defense_time)
        # STATISTICS
        _, sim_pred = preds.max(2)
        for pred, gt in zip(sim_pred, target):
            compare = [1 if int(pred[i]) == int(gt[i]) else 0 for i in range(8)]
            if sum(compare) == 8:
                n_correct += 1

        correct += n_correct
        batch_time.update(time.time() - end)
        acc.update(n_correct / float(inputs.size(0)), float(inputs.size(0)))
        end = time.time()

        if i % config.print_freq == 0:
            print('Iter: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'ACC  {acc.val:.4f} ({acc.avg:.4f})\t' .format(
                             i, len(dataloader),
                             batch_time=batch_time,
                             data_time=data_time, acc=acc))
    print('DEFENSE FPS:', 65000/defense_time)
    print(correct/acc.count)
    return acc.avg


def load_data(args, sub_item):
    img_dir = '/data/private_data/jf_private/yizhi_data/LP_AD/CW/ADV/'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_dataset = CCPDdataset(img_dir, sub_item=sub_item, debug=False, train=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                             drop_last=False)
    return val_loader


def main():
    if 'vgg' in args.weight:
        LPR = VGG(config.nclass)
        sub_item = ['vgg']
    elif 'res' in args.weight:
        LPR = ResNet()
        sub_item = ['res']
    elif 'dense' in args.weight:
        LPR = DenseNet121()
        sub_item = ['dense']
    else:
        LPR = BotNet50()
        sub_item = ['trans']

    defense = args.defense
    if defense == 1:
        d_block = UNet()
        d_block.load_state_dict(torch.load(args.defense_weight, map_location='cpu')['model'])
        d_block.cuda().eval()
    else:
        d_block = None

    print('loading pretrained model from %s' % args.weight)
    LPR.load_state_dict(torch.load(args.weight, map_location='cpu')['state_dict'])
    LPR = LPR.cuda()
    val_loader = load_data(args, sub_item)
    acc = val(LPR, val_loader, d_block=d_block)
    print(acc)


if __name__ == '__main__':
    main()
