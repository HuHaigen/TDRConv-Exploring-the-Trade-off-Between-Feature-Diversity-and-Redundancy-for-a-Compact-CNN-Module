from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TDRConv import TDRConv


norm_mean, norm_var = 0.0, 1.0


def conv3x3_test(in_planes, out_planes, kernel_size=(3,3), padding=(1,1), bias=False, stride=(1, 1)):
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)
    kernel_size = kernel_size[0]
    stride = stride[0]
    padding = padding[0]
    return TDRConv(in_planes, out_planes, kernel_size=3, stride=stride)



def conv1x1_test(in_planes, out_planes, kernel_size=1, padding=1, bias=False, stride=(1, 1)):
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)
    stride = stride[0]
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), bias=False)

def conv3x3_similarity(in_planes, out_planes, kernel_size=3, padding=1, bias=False, stride=(1, 1)):
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.over_all_relu = nn.ReLU(inplace=True)

        # 起始层
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer0_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # layer1.ResBottleneck0
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer3_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck0.downsample
        self.layer4_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck1
        self.layer5_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer6_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer7_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck2
        self.layer8_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer8_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer9_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer9_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer10_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer10_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck0
        self.layer11_conv = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer12_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer13_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer13_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2.ResBottleneck0.downsample
        self.layer14_conv = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer14_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck1
        self.layer15_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer15_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer16_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer16_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer17_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck2
        self.layer18_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer18_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer19_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer19_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer20_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer20_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck3
        self.layer21_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer21_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer22_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer23_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0
        self.layer24_conv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer24_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer25_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer25_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer26_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer26_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0.downsample
        self.layer27_conv = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer27_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck1
        self.layer28_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer28_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer29_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer29_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer30_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer30_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck2
        self.layer31_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer31_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer32_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer33_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck3
        self.layer34_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer34_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer35_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer36_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer36_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck4
        self.layer37_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer38_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer38_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer39_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer39_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck5
        self.layer40_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer40_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer41_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer42_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer42_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0
        self.layer43_conv = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer44_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer44_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer45_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer45_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0.downsample
        self.layer46_conv = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer46_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck1
        self.layer47_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer48_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer48_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer49_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer49_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck2
        self.layer50_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer50_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer51_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer52_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer52_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(2048, num_classes))
        ]))

        #
        # self.downsample_layer6_residual = nn.Conv2d(64 exp=2, 128, kernel_size=(1, 1),stride=(2, 2), bias=False).cuda()
        # self.downsample_layer6_residual_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
        #                                                       track_running_stats=True).cuda()
        #
        # # layer3.BasicBlock0.downsample
        # self.downsample_layer15_residual = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer15_residual_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
        #                                                        affine=True, track_running_stats=True).cuda()
        #
        # # layer4.BasicBlock0.downsample
        # self.downsample_layer28_residual = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer28_residual_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        out_fc = self.classifier(y)
        return out_fc


class ResNet_noClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_noClassifier, self).__init__()
        self.over_all_relu = nn.ReLU(inplace=True)

        # 起始层
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer0_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # layer1.ResBottleneck0
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer3_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck0.downsample
        self.layer4_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck1
        self.layer5_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer6_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer7_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck2
        self.layer8_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer8_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer9_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer9_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer10_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer10_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck0
        self.layer11_conv = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer12_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer12_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer13_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer13_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2.ResBottleneck0.downsample
        self.layer14_conv = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer14_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck1
        self.layer15_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer15_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer16_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer16_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer17_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck2
        self.layer18_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer18_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer19_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer19_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer20_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer20_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck3
        self.layer21_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer21_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer22_conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer23_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0
        self.layer24_conv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer24_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer25_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer25_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer26_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer26_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0.downsample
        self.layer27_conv = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer27_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck1
        self.layer28_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer28_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer29_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer29_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer30_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer30_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck2
        self.layer31_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer31_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer32_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer33_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck3
        self.layer34_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer34_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer35_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer36_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer36_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck4
        self.layer37_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer38_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer38_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer39_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer39_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck5
        self.layer40_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer40_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer41_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer42_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer42_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0
        self.layer43_conv = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer44_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer44_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer45_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer45_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0.downsample
        self.layer46_conv = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer46_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck1
        self.layer47_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer48_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer48_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer49_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer49_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck2
        self.layer50_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer50_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer51_conv = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer52_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer52_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # self.classifier = nn.Sequential(OrderedDict([
        #     ('linear1', nn.Linear(2048, num_classes))
        # ]))

        #
        # self.downsample_layer6_residual = nn.Conv2d(64 exp=2, 128, kernel_size=(1, 1),stride=(2, 2), bias=False).cuda()
        # self.downsample_layer6_residual_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
        #                                                       track_running_stats=True).cuda()
        #
        # # layer3.BasicBlock0.downsample
        # self.downsample_layer15_residual = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer15_residual_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
        #                                                        affine=True, track_running_stats=True).cuda()
        #
        # # layer4.BasicBlock0.downsample
        # self.downsample_layer28_residual = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer28_residual_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        # out_fc = self.classifier(y)
        out_fc = y
        return out_fc


class ResNet_test(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_test, self).__init__()
        self.over_all_relu = nn.ReLU(inplace=True)

        # 起始层
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer0_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # layer1.ResBottleneck0
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer3_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck0.downsample
        self.layer4_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck1
        self.layer5_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer6_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer7_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck2
        self.layer8_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer8_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer9_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer9_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer10_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer10_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck0
        self.layer11_conv = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer12_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer12_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer13_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer13_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2.ResBottleneck0.downsample
        self.layer14_conv = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer14_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck1
        self.layer15_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer15_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer16_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer16_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer17_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck2
        self.layer18_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer18_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer19_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer19_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer20_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer20_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck3
        self.layer21_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer21_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer22_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer23_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0
        self.layer24_conv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer24_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer25_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer25_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer26_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer26_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0.downsample
        self.layer27_conv = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer27_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck1
        self.layer28_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer28_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer29_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer29_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer30_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer30_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck2
        self.layer31_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer31_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer32_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer33_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck3
        self.layer34_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer34_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer35_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer36_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer36_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck4
        self.layer37_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer38_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer38_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer39_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer39_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck5
        self.layer40_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer40_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer41_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer42_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer42_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0
        self.layer43_conv = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer44_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer44_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer45_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer45_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0.downsample
        self.layer46_conv = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer46_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck1
        self.layer47_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer48_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer48_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer49_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer49_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck2
        self.layer50_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer50_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer51_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer52_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer52_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(2048, num_classes))
        ]))

        #
        # self.downsample_layer6_residual = nn.Conv2d(64 exp=2, 128, kernel_size=(1, 1),stride=(2, 2), bias=False).cuda()
        # self.downsample_layer6_residual_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
        #                                                       track_running_stats=True).cuda()
        #
        # # layer3.BasicBlock0.downsample
        # self.downsample_layer15_residual = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer15_residual_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
        #                                                        affine=True, track_running_stats=True).cuda()
        #
        # # layer4.BasicBlock0.downsample
        # self.downsample_layer28_residual = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer28_residual_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('0device..')
        # print(x.device)
        # print(next(self.layer0_conv.parameters()).device)
        # print(next(self.layer0_norm.parameters()).device)
        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        out_fc = self.classifier(y)
        return out_fc
    
    def forward2(self, x):
        s_list = []
        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        # layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))
        y2, s2 = self.layer2_conv(layer1_out)
        s_list.append(s2)
        layer2_out = self.over_all_relu(self.layer2_norm(y2))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        # layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        y6, s6 = self.layer6_conv(layer5_out)
        s_list.append(s6)
        layer6_out = self.over_all_relu(self.layer6_norm(y6))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        # layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        y9, s9 = self.layer9_conv(layer8_out)
        s_list.append(s9)
        layer9_out = self.over_all_relu(self.layer9_norm(y9))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        # layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        y12, s12 = self.layer12_conv(layer11_out)
        s_list.append(s12)
        layer12_out = self.over_all_relu(self.layer12_norm(y12))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        # layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        y16, s16 = self.layer16_conv(layer15_out)
        s_list.append(s16)
        layer16_out = self.over_all_relu(self.layer16_norm(y16))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        # layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        y19, s19 = self.layer19_conv(layer18_out)
        s_list.append(s19)
        layer19_out = self.over_all_relu(self.layer19_norm(y19))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        # layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        y22, s22 = self.layer22_conv(layer21_out)
        s_list.append(s22)
        layer22_out = self.over_all_relu(self.layer22_norm(y22))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        # layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        y25, s25 = self.layer25_conv(layer24_out)
        s_list.append(s25)
        layer25_out = self.over_all_relu(self.layer25_norm(y25))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        # layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        y29, s29 = self.layer29_conv(layer28_out)
        s_list.append(s29)
        layer29_out = self.over_all_relu(self.layer29_norm(y29))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        # layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        y32, s32 = self.layer32_conv(layer31_out)
        s_list.append(s32)
        layer32_out = self.over_all_relu(self.layer32_norm(y32))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        # layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        y35, s35 = self.layer35_conv(layer34_out)
        s_list.append(s35)
        layer35_out = self.over_all_relu(self.layer35_norm(y35))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        # layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        y38, s38 = self.layer38_conv(layer37_out)
        s_list.append(s38)
        layer38_out = self.over_all_relu(self.layer38_norm(y38))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        # layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        y41, s41 = self.layer41_conv(layer40_out)
        s_list.append(s41)
        layer41_out = self.over_all_relu(self.layer41_norm(y41))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        # layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        y44, s44 = self.layer44_conv(layer43_out)
        s_list.append(s44)
        layer44_out = self.over_all_relu(self.layer44_norm(y44))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        # layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        y48, s48 = self.layer48_conv(layer47_out)
        s_list.append(s48)
        layer48_out = self.over_all_relu(self.layer48_norm(y48))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        # layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        y51, s51 = self.layer51_conv(layer50_out)
        s_list.append(s51)
        layer51_out = self.over_all_relu(self.layer51_norm(y51))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        out_fc = self.classifier(y)
        return out_fc,s_list

class ResNet_test2(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_test2, self).__init__()
        self.over_all_relu = nn.ReLU(inplace=True)

        # 起始层
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer0_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # layer1.ResBottleneck0
        self.layer1_conv = conv1x1_test(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer3_conv = conv1x1_test(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck0.downsample
        self.layer4_conv = conv1x1_test(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck1
        self.layer5_conv = conv1x1_test(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer6_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer7_conv = conv1x1_test(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck2
        self.layer8_conv = conv1x1_test(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer8_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer9_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer9_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer10_conv = conv1x1_test(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer10_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck0
        self.layer11_conv = conv1x1_test(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer12_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer12_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer13_conv = conv1x1_test(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer13_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2.ResBottleneck0.downsample
        self.layer14_conv = conv1x1_test(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer14_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck1
        self.layer15_conv = conv1x1_test(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer15_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer16_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer16_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer17_conv = conv1x1_test(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck2
        self.layer18_conv = conv1x1_test(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer18_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer19_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer19_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer20_conv = conv1x1_test(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer20_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck3
        self.layer21_conv = conv1x1_test(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer21_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer22_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer23_conv = conv1x1_test(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0
        self.layer24_conv = conv1x1_test(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer24_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer25_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer25_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer26_conv = conv1x1_test(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer26_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0.downsample
        self.layer27_conv = conv1x1_test(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer27_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck1
        self.layer28_conv = conv1x1_test(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer28_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer29_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer29_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer30_conv = conv1x1_test(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer30_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck2
        self.layer31_conv = conv1x1_test(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer31_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer32_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer33_conv = conv1x1_test(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck3
        self.layer34_conv = conv1x1_test(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer34_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer35_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer36_conv = conv1x1_test(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer36_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck4
        self.layer37_conv = conv1x1_test(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer38_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer38_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer39_conv = conv1x1_test(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer39_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck5
        self.layer40_conv = conv1x1_test(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer40_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer41_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer42_conv = conv1x1_test(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer42_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0
        self.layer43_conv = conv1x1_test(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer44_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer44_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer45_conv = conv1x1_test(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer45_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0.downsample
        self.layer46_conv = conv1x1_test(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer46_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck1
        self.layer47_conv = conv1x1_test(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer48_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer48_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer49_conv = conv1x1_test(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer49_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck2
        self.layer50_conv = conv1x1_test(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer50_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer51_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer52_conv = conv1x1_test(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer52_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(2048, num_classes))
        ]))

        #
        # self.downsample_layer6_residual = nn.Conv2d(64 exp=2, 128, kernel_size=(1, 1),stride=(2, 2), bias=False).cuda()
        # self.downsample_layer6_residual_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
        #                                                       track_running_stats=True).cuda()
        #
        # # layer3.BasicBlock0.downsample
        # self.downsample_layer15_residual = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer15_residual_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
        #                                                        affine=True, track_running_stats=True).cuda()
        #
        # # layer4.BasicBlock0.downsample
        # self.downsample_layer28_residual = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer28_residual_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s_list = []
        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        # layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))
        y2, s2 = self.layer2_conv(layer1_out)
        s_list.append(s2)
        layer2_out = self.over_all_relu(self.layer2_norm(y2))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        # layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        y6, s6 = self.layer6_conv(layer5_out)
        s_list.append(s6)
        layer6_out = self.over_all_relu(self.layer6_norm(y6))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        # layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        y9, s9 = self.layer9_conv(layer8_out)
        s_list.append(s9)
        layer9_out = self.over_all_relu(self.layer9_norm(y9))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        # layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        y12, s12 = self.layer12_conv(layer11_out)
        s_list.append(s12)
        layer12_out = self.over_all_relu(self.layer12_norm(y12))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        # layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        y16, s16 = self.layer16_conv(layer15_out)
        s_list.append(s16)
        layer16_out = self.over_all_relu(self.layer16_norm(y16))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        # layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        y19, s19 = self.layer19_conv(layer18_out)
        s_list.append(s19)
        layer19_out = self.over_all_relu(self.layer19_norm(y19))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        # layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        y22, s22 = self.layer22_conv(layer21_out)
        s_list.append(s22)
        layer22_out = self.over_all_relu(self.layer22_norm(y22))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        # layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        y25, s25 = self.layer25_conv(layer24_out)
        s_list.append(s25)
        layer25_out = self.over_all_relu(self.layer25_norm(y25))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        # layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        y29, s29 = self.layer29_conv(layer28_out)
        s_list.append(s29)
        layer29_out = self.over_all_relu(self.layer29_conv(y29))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        # layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        y32, s32 = self.layer32_conv(layer31_out)
        s_list.append(s32)
        layer32_out = self.over_all_relu(self.layer32_norm(y32))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        # layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        y35, s35 = self.layer35_conv(layer34_out)
        s_list.append(s35)
        layer35_out = self.over_all_relu(self.layer35_norm(y35))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        # layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        y38, s38 = self.layer38_conv(layer37_out)
        s_list.append(s38)
        layer38_out = self.over_all_relu(self.layer38_norm(y38))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        # layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        y41, s41 = self.layer41_conv(layer40_out)
        s_list.append(s41)
        layer41_out = self.over_all_relu(self.layer41_norm(y41))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        # layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        y44, s44 = self.layer44_conv(layer43_out)
        s_list.append(s44)
        layer44_out = self.over_all_relu(self.layer44_norm(y44))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        # layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        y48, s48 = self.layer48_conv(layer47_out)
        s_list.append(s48)
        layer48_out = self.over_all_relu(self.layer48_norm(y48))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        # layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        y51, s51 = self.layer51_conv(layer50_out)
        s_list.append(s51)
        layer51_out = self.over_all_relu(self.layer51_norm(y51))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        out_fc = self.classifier(y)
        return out_fc,s_list

    def forward2(self, x):

        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        out_fc = self.classifier(y)
        return out_fc


class ResNet_test_noClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_test_noClassifier, self).__init__()
        self.over_all_relu = nn.ReLU(inplace=True)

        # 起始层
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer0_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # layer1.ResBottleneck0
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer3_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck0.downsample
        self.layer4_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck1
        self.layer5_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer5_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer6_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer7_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer1.ResBottleneck2
        self.layer8_conv = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer8_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer9_conv = conv3x3_test(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer9_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer10_conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer10_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck0
        self.layer11_conv = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer12_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer12_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer13_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer13_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # layer2.ResBottleneck0.downsample
        self.layer14_conv = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer14_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck1
        self.layer15_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer15_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer16_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer16_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer17_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer17_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck2
        self.layer18_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer18_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer19_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer19_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer20_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer20_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer2.ResBottleneck3
        self.layer21_conv = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer21_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer22_conv = conv3x3_test(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer23_conv = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer23_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0
        self.layer24_conv = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer24_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer25_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer25_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer26_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer26_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck0.downsample
        self.layer27_conv = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer27_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck1
        self.layer28_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer28_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer29_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer29_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer30_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer30_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck2
        self.layer31_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer31_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer32_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer33_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck3
        self.layer34_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer34_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer35_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer36_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer36_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck4
        self.layer37_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer38_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer38_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer39_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer39_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer3.ResBottleneck5
        self.layer40_conv = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer40_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer41_conv = conv3x3_test(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer42_conv = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer42_norm = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0
        self.layer43_conv = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer44_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer44_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer45_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer45_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck0.downsample
        self.layer46_conv = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer46_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck1
        self.layer47_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer48_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer48_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer49_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer49_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # layer4.ResBottleneck2
        self.layer50_conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer50_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer51_conv = conv3x3_test(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer52_conv = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer52_norm = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # self.classifier = nn.Sequential(OrderedDict([
        #     ('linear1', nn.Linear(2048, num_classes))
        # ]))

        #
        # self.downsample_layer6_residual = nn.Conv2d(64 exp=2, 128, kernel_size=(1, 1),stride=(2, 2), bias=False).cuda()
        # self.downsample_layer6_residual_norm = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
        #                                                       track_running_stats=True).cuda()
        #
        # # layer3.BasicBlock0.downsample
        # self.downsample_layer15_residual = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer15_residual_norm = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,
        #                                                        affine=True, track_running_stats=True).cuda()
        #
        # # layer4.BasicBlock0.downsample
        # self.downsample_layer28_residual = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        # self.downsample_layer28_residual_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        layer0_out = self.over_all_relu(self.layer0_norm(self.layer0_conv(x)))
        pool0_out = self.pool0(layer0_out)

        layer1_out = self.over_all_relu(self.layer1_norm(self.layer1_conv(pool0_out)))
        layer2_out = self.over_all_relu(self.layer2_norm(self.layer2_conv(layer1_out)))

        layer3_out = self.layer3_norm(self.layer3_conv(layer2_out))
        # layer1.ResBottleneck0.downsample
        layer4_out = self.layer4_norm(self.layer4_conv(pool0_out))
        layer4_residual = self.over_all_relu(layer4_out + layer3_out)

        layer5_out = self.over_all_relu(self.layer5_norm(self.layer5_conv(layer4_residual)))
        layer6_out = self.over_all_relu(self.layer6_norm(self.layer6_conv(layer5_out)))
        layer7_out = self.layer7_norm(self.layer7_conv(layer6_out))

        layer7_residual = self.over_all_relu(layer4_residual + layer7_out)

        layer8_out = self.over_all_relu(self.layer8_norm(self.layer8_conv(layer7_residual)))
        layer9_out = self.over_all_relu(self.layer9_norm(self.layer9_conv(layer8_out)))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))

        layer10_residual = self.over_all_relu(layer7_residual + layer10_out)

        layer11_out = self.over_all_relu(self.layer11_norm(self.layer11_conv(layer10_residual)))
        layer12_out = self.over_all_relu(self.layer12_norm(self.layer12_conv(layer11_out)))
        layer13_out = self.layer13_norm(self.layer13_conv(layer12_out))

        # layer2.ResBottleneck0.downsample
        layer14_out = self.layer14_norm(self.layer14_conv(layer10_residual))
        layer14_residual = self.over_all_relu(layer14_out + layer13_out)

        layer15_out = self.over_all_relu(self.layer15_norm(self.layer15_conv(layer14_residual)))
        layer16_out = self.over_all_relu(self.layer16_norm(self.layer16_conv(layer15_out)))
        layer17_out = self.layer17_norm(self.layer17_conv(layer16_out))
        layer17_residual = self.over_all_relu(layer14_residual + layer17_out)

        layer18_out = self.over_all_relu(self.layer18_norm(self.layer18_conv(layer17_residual)))
        layer19_out = self.over_all_relu(self.layer19_norm(self.layer19_conv(layer18_out)))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer20_residual = self.over_all_relu(layer17_residual + layer20_out)

        layer21_out = self.over_all_relu(self.layer21_norm(self.layer21_conv(layer20_residual)))
        layer22_out = self.over_all_relu(self.layer22_norm(self.layer22_conv(layer21_out)))
        layer23_out = self.layer23_norm(self.layer23_conv(layer22_out))
        layer23_residual = self.over_all_relu(layer20_residual + layer23_out)

        # layer3.ResBottleneck0
        layer24_out = self.over_all_relu(self.layer24_norm(self.layer24_conv(layer23_residual)))
        layer25_out = self.over_all_relu(self.layer25_norm(self.layer25_conv(layer24_out)))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))

        # layer3.ResBottleneck0.downsample
        layer27_out = self.layer27_norm(self.layer27_conv(layer23_residual))
        layer27_residual = self.over_all_relu(layer26_out + layer27_out)

        # layer3.ResBottleneck1
        layer28_out = self.over_all_relu(self.layer28_norm(self.layer28_conv(layer27_residual)))
        layer29_out = self.over_all_relu(self.layer29_norm(self.layer29_conv(layer28_out)))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        layer30_residual = self.over_all_relu(layer27_residual + layer30_out)

        # layer3.ResBottleneck2
        layer31_out = self.over_all_relu(self.layer31_norm(self.layer31_conv(layer30_residual)))
        layer32_out = self.over_all_relu(self.layer32_norm(self.layer32_conv(layer31_out)))
        layer33_out = self.layer33_norm(self.layer33_conv(layer32_out))
        layer33_residual = self.over_all_relu(layer30_residual + layer33_out)

        # layer3.ResBottleneck3
        layer34_out = self.over_all_relu(self.layer34_norm(self.layer34_conv(layer33_residual)))
        layer35_out = self.over_all_relu(self.layer35_norm(self.layer35_conv(layer34_out)))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        layer36_residual = self.over_all_relu(layer33_residual + layer36_out)

        # layer3.ResBottleneck4
        layer37_out = self.over_all_relu(self.layer37_norm(self.layer37_conv(layer36_residual)))
        layer38_out = self.over_all_relu(self.layer38_norm(self.layer38_conv(layer37_out)))
        layer39_out = self.layer39_norm(self.layer39_conv(layer38_out))
        layer39_residual = self.over_all_relu(layer36_residual + layer39_out)

        # layer3.ResBottleneck5
        layer40_out = self.over_all_relu(self.layer40_norm(self.layer40_conv(layer39_residual)))
        layer41_out = self.over_all_relu(self.layer41_norm(self.layer41_conv(layer40_out)))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        layer42_residual = self.over_all_relu(layer39_residual + layer42_out)

        # layer4.ResBottleneck0
        layer43_out = self.over_all_relu(self.layer43_norm(self.layer43_conv(layer42_residual)))
        layer44_out = self.over_all_relu(self.layer44_norm(self.layer44_conv(layer43_out)))
        layer45_out = self.layer45_norm(self.layer45_conv(layer44_out))

        # layer4.ResBottleneck0.downsample
        layer46_out = self.layer46_norm(self.layer46_conv(layer42_residual))
        layer46_residual = self.over_all_relu(layer46_out + layer45_out)

        # layer4.ResBottleneck1
        layer47_out = self.over_all_relu(self.layer47_norm(self.layer47_conv(layer46_residual)))
        layer48_out = self.over_all_relu(self.layer48_norm(self.layer48_conv(layer47_out)))
        layer49_out = self.layer49_norm(self.layer49_conv(layer48_out))
        layer49_residual = self.over_all_relu(layer46_residual + layer49_out)

        # layer4.ResBottleneck2
        layer50_out = self.over_all_relu(self.layer50_norm(self.layer50_conv(layer49_residual)))
        layer51_out = self.over_all_relu(self.layer51_norm(self.layer51_conv(layer50_out)))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        layer52_residual = self.over_all_relu(layer49_residual + layer52_out)
        y = self.avgpool(layer52_residual)
        y = y.view(y.size(0), -1)
        # out_fc = self.classifier(y)
        out_fc = y
        return out_fc





def resnet_50(num_class):
    return ResNet(num_class)


def resnet_50_tdrc(num_class):
    return ResNet_test(num_class)


if __name__ == '__main__':
    model = resnet_50(10)
    print(model)
