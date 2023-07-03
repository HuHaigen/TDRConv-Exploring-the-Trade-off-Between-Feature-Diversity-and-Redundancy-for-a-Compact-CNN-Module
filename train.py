from datetime import datetime
import sys
from pathlib import Path
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from collections import Counter
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.optim.lr_scheduler as lr_scheduler
from model.resnet56model import resnet_56
from model.vgg16model_1fc import vgg_16_1fc,vgg_16_1fc_tdrc
from model.resnet50model import resnet_50, resnet_50_tdrc
from model.shufflenet_v2 import shufflenet_v2_x1_0
import os
import argparse
from utils.likelyhood_utils import get_model, get_count_dict
from utils.net_utils import get_loaders, AverageMeter, accuracy,count_params
from datetime import datetime
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--data-dir',
    type=str,
    # default='~/data/ImageNet',
    default='~/Deming/ghostnet_cifar10-master/data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    # default='imagenet',
    default='cifar10',
    choices=('cifar10', 'cifar100', 'imagenet'),
    help='dataset')
parser.add_argument(
    '--lr',
    default=0.1,
    type=float,
    help='initial learning rate')


parser.add_argument(
    '--lrf',
    default=0.0001,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--lr-decay-step',
    default='5,10',
    type=str,
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    # default='resnet_50_imagenet',
    default='none',
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--milestones',
    type=list,
    # default=[30, 60,90, 120, 150,180,210,240,270,290],
    default=[30, 55,75, 120, 150],
    # default=[92,136],
    help='optimizer milestones')
parser.add_argument(
    '--train-batch-size',
    type=int,
    default=256,
    help='Batch size for training.')
parser.add_argument(
    '--eval-batch-size',
    type=int,
    default=256,
    help='Batch size for validation.')

parser.add_argument(
    '--epoch',
    type=int,
    default=130,
    help='epoch')

parser.add_argument(
    '--compress-rate',
    type=float,
    default=0.7,
    help='The architecture to prune')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_50',
    choices=(
        'vgg_16_1fc',  'resnet_56', 'resnet_56_tdrc','resnet_50', 'resnet_50_tdrc','vgg_16_1fc_tdrc'
        ,'shufflenet_v2_x1_0'),
    help='The architecture to prune')


parser.add_argument(
    '--mark',
    type=str,
    default='None',
    help='mark id')


parser.add_argument(
    '--simloss',
    type=str,
    default='no',
    help='if add similarity loss')

parser.add_argument(
    '--lr-cosin',
    type=str,
    default='no',
    help='if use cosin strategy')


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))

if args.dataset == 'cifar100':
    args.num_class = 100
elif args.dataset == 'imagenet':
    args.num_class = 1000
    args.data_dir == '~/data/ImageNet'
else:
    args.num_class = 10
    args.data_dir == '~/data/cifar10'
    args.milestones == [92,136]
    args.epoch = 190

print(vars(args))

project_root_path = os.path.abspath(os.path.dirname(__file__))
trainloader, testloader = get_loaders(args.dataset, args.data_dir, args.train_batch_size, args.eval_batch_size,
                                      args.arch)

# Model
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if sys.platform == 'linux':
    baseline_dir = './MyDrive/'
else:
    baseline_dir = os.path.join(project_root_path, 'pth')

if not Path(baseline_dir).exists():
    os.mkdir(baseline_dir)


time_str = datetime.strftime(datetime.now(),'%m-%d_%H-%M')

# Training
def train_baseline():
    if args.resume != "none":
        print('checkpoint %s exists,train from checkpoint' % args.resume)
        save_path = os.path.join(baseline_dir, args.resume + '.pth')
        # model_state = torch.load(args.resume, map_location=device)
        model_state = get_model(args.resume)
        # print(model_state)
        net = model_state['net']
        current_model_best_acc = model_state['best_prec1']
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        # optimizer.load_state_dict(model_state['optimizer'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()
        try:
            start_epoch = model_state['epoch']
        except KeyError:
            start_epoch = 0
        end_epoch = start_epoch + 300
    else:
        if args.mark == 'None':
            args.mark = time_str
        save_path = os.path.join(
            baseline_dir, args.arch + '_' + args.dataset +'_'+args.mark+ '.pth')

        current_model_best_acc = 0
        # net = VGG(args.num_class)
        # net = eval(args.arch)(num_classes=args.num_class)
        net = eval(args.arch)(num_class=args.num_class)
        print(net)
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        start_epoch = 0
        end_epoch = args.epoch
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    print('max_eopch: ',end_epoch)
    # gpusetting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = torch.nn.DataParallel(net, device_ids=[0])
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    # 输入数据集的size
    # input_size = 224
    if args.dataset == 'imagenet':
        input_size = 224
    elif args.dataset =='cifar10':
        input_size = 32
    # params and flops
    count_params(model=net,input_size=input_size)
    # net = net.cuda(0)

    cudnn.benchmark = True
    print('\nEpoch: %d' % start_epoch)
    count_dict = {}
    criterion = torch.nn.CrossEntropyLoss()
    filter_count = Counter()
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    if args.lr_cosin == 'yes':
        print('using the cosine strategy...')
        pg = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    print("Init lr: ", optimizer.defaults['lr'])
    for epoch in range(start_epoch + 1, end_epoch):
        # if epoch in [30, 60, 90, 120, 150]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        # adjust_learning_rate(optimizer,epoch,args)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        # temp_tensor = torch.zeros(512,512)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            with torch.cuda.device(0):
                # with torch.cuda.device(0):
                inputs = inputs.to(device)
                targets = targets.to(device)
                # inputs = inputs.cuda(0)
                # targets = targets.cuda(0)
                optimizer.zero_grad()
                outputs = net(inputs)
                # np.linalg.norm(net.layer0_conv.weight.grad.view(64, -1).cpu().numpy(),axis=0)

                loss = criterion(outputs, targets)
                loss.backward()
                if args.resume == 'none':
                    count_dict = get_count_dict(
                        net, args.compress_rate, count_dict)

                optimizer.step()

                # print(f"epoch {epoch}, current batch {batch_idx}, loss {loss.item()}")
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # print(batch_idx, len(trainloader),
                #       ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #       % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print("epoch %d lr：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        top1 = AverageMeter()
        top5 = AverageMeter()
        net.eval()
        num_iterations = len(testloader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = inputs.cuda(0), targets.cuda(0)
                outputs = net(inputs)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

            print(
                'Epoch[{0}]({1}/{2}): '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iterations, top1=top1, top5=top5))

        if (top1.avg.item() > current_model_best_acc):
            # print("save model")
            current_model_best_acc = top1.avg.item()
            model_state = {
                'net': net,
                'best_prec1': current_model_best_acc,
                'epoch': epoch,
                'optimizer': optimizer,
                'counter_dict': count_dict
            }
            print("save model...")
            torch.save(net.state_dict(), save_path)
            # if args.arch in ['shufflenet_v2_x1_0','mobilenetv2','vgg_16_1fc_test','']:
            #     print("save model...")
            #     torch.save(net.state_dict(), save_path)
            # else:
            #     print("save model.")
            #     torch.save(model_state, save_path)
            

    print("=>Best accuracy {:.3f}".format(model_state['best_prec1']))



def train_baseline_addReg():
    print('train_baseline_addReg')
    if args.resume != "none":
        print('checkpoint %s exists,train from checkpoint' % args.resume)
        save_path = os.path.join(baseline_dir, args.resume + '.pth')
        # model_state = torch.load(args.resume, map_location=device)
        model_state = get_model(args.resume)
        # print(model_state)
        net = model_state['net']
        current_model_best_acc = model_state['best_prec1']
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        # optimizer.load_state_dict(model_state['optimizer'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()
        try:
            start_epoch = model_state['epoch']
        except KeyError:
            start_epoch = 0
        end_epoch = start_epoch + 300
    else:
        if args.mark == 'None':
            args.mark = time_str
        save_path = os.path.join(
            baseline_dir, args.arch + '_' + args.dataset +'_'+args.mark+ '.pth')

        current_model_best_acc = 0
        # net = VGG(args.num_class)
        net = eval(args.arch)(args.num_class)
        print(net)
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)
        start_epoch = 0
        end_epoch = args.epoch
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    print('max_eopch: ',end_epoch)
    # gpusetting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = torch.nn.DataParallel(net, device_ids=[0])
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    # net = net.cuda(0)

    cudnn.benchmark = True
    print('\nEpoch: %d' % start_epoch)
    count_dict = {}
    criterion = torch.nn.CrossEntropyLoss()
    filter_count = Counter()
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    print("Init lr: ", optimizer.defaults['lr'])
    if(args.simloss !='no'):
        print('use simloss...')
    for epoch in range(start_epoch + 1, end_epoch):
        # if epoch in [30, 60, 90, 120, 150]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        # adjust_learning_rate(optimizer,epoch,args)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        # temp_tensor = torch.zeros(512,512)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            with torch.cuda.device(0):
                # with torch.cuda.device(0):
                inputs = inputs.to(device)
                targets = targets.to(device)
                # inputs = inputs.cuda(0)
                # targets = targets.cuda(0)
                optimizer.zero_grad()
                # print('device..')
                # print(inputs.device)
                # print(next(net.parameters()).device)
                outputs,s_list = net(inputs)
                # np.linalg.norm(net.layer0_conv.weight.grad.view(64, -1).cpu().numpy(),axis=0)
                # 相似度增大项
                # sim_up = (-1.*s_list[0]) +  (-1.*s_list[1]) + (-1.*s_list[2]) 
                # sim_up = (s_list[0]) +  (s_list[1]) + s_list[2] +s_list[3]+s_list[4]
                # sim_up = (s_list[0]) +  (s_list[1]) + s_list[2] 
                sim_up = 0.
                # print("list0",s_list[0])
                # sim_down = s_list[-1] + s_list[-2] + s_list[-3]+ s_list[-4]+s_list[-5]
                # sim_down = s_list[-1] + s_list[-2] + s_list[-3]
                sim_down = 0.
                # lamd = torch.tensor(1e-1)
                lamd = torch.tensor(0.5*1e-1)
                reg = torch.mean(lamd*(sim_up + sim_down))
                # print("reg",reg)
                # print("reg",reg)
                # print("lossc",criterion(outputs, targets))
                # for i,s in enumerate(s_list,start = 1):
                loss1 = criterion(outputs, targets)
                print('loss1',loss1)
                print('reg',reg)
                loss = loss1
                if(args.simloss !='no'):
                    loss = loss + reg
                    # print('use simloss...')
                
                loss.backward()
                if args.resume == 'none':
                    count_dict = get_count_dict(
                        net, args.compress_rate, count_dict)

                optimizer.step()

                # print(f"epoch {epoch}, current batch {batch_idx}, loss {loss.item()}")
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # print(batch_idx, len(trainloader),
                #       ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #       % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print("epoch %d lr：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        top1 = AverageMeter()
        top5 = AverageMeter()
        net.eval()
        num_iterations = len(testloader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = inputs.cuda(0), targets.cuda(0)
                outputs,s_list = net(inputs)
                s_list2=[]
                for i, val in enumerate(s_list):
                    s_list2.append(val.mean(-1).cpu().numpy())
                    
                print("similarity_list",s_list2)

                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

            print(
                'Epoch[{0}]({1}/{2}): '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iterations, top1=top1, top5=top5))

        if (top1.avg.item() > current_model_best_acc):
            print("save model")
            current_model_best_acc = top1.avg.item()
            model_state = {
                'net': net,
                'best_prec1': current_model_best_acc,
                'epoch': epoch,
                'optimizer': optimizer,
                'counter_dict': count_dict
            }

            torch.save(model_state, save_path)

    print("=>Best accuracy {:.3f}".format(model_state['best_prec1']))



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    print("train start time:",time_str)
    train_baseline()
    # train_baseline_addReg()
    # drop_then_train()
