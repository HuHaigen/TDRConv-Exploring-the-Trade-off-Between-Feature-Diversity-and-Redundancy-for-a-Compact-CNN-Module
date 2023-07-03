import glob
import os
import sys

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
from thop import profile
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torchstat import stat
import numpy as np

from model.resnet56model import resnet_56

import argparse

from utils.likelyhood_utils import get_model
from utils.net_utils import get_loaders, AverageMeter, accuracy
from torch import nn

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--model_name',
    type=str,
    default='resnet_50_imagenet_pruned',
    help='dataset path')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/data/ImageNet',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='imagenet',
    help='dataset')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size for validation.')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_50',
    choices=('resnet_56','vgg_16_1fc'),
    help='The architecture to prune')

args = parser.parse_args()
args.train_batch_size = 32
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

if args.dataset == 'cifar100':
    args.num_class = 100
elif args.dataset == 'imagenet':
    args.num_class = 1000
else:
    args.num_class = 10

device = torch.device('cuda')

best_acc = 0  # best test accuracy

trainloader,testloader = get_loaders(args.dataset, args.data_dir,args.train_batch_size,args.eval_batch_size,args.arch)


def test():

    model_state = get_model(args.model_name)
    # try:
    #     cfg = model_state['cfg']
    # except KeyError:
    #     cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
    # net = VGG(10)

    # if 'net' in model_state.keys():
    #     net.load_state_dict(model_state['state_dict'])
    # else:
    #     net.load_state_dict(model_state)
    net = model_state['net']

    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

    input = torch.randn(1,3, 224, 224,device='cuda:1')
    flops, params = profile(net.module, inputs=(input, ))
    print('flops: %.2fM' % (flops/1e6))
    batch_count = len(testloader)
    recall_macro = 0
    acc = 0
    recall_micro = 0
    precision_macro = 0
    precision_micro = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(0), targets.to(0)

            outputs = net(inputs)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
        end_time = time.time()
        print(end_time - start_time)
    print(top1.avg.item(),top5.avg.item())
    # print("recall_macro = ",round(recall_macro/batch_count,3))
    # print("acc = ", round(acc / batch_count,3))
    # print("recall_micro = ", round(recall_micro / batch_count,3))
    # print("precision_macro = ", round(precision_macro / batch_count,3))
    # print("precision_micro = ", round(precision_micro / batch_count,3))


if __name__ == '__main__':

    test()



