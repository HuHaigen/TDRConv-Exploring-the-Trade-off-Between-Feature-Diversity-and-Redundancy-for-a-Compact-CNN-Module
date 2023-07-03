from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TDRConv import TDRConv

norm_mean, norm_var = 0.0, 1.0


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_test(in_planes, out_planes, stride=1):
    return TDRConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)





class ResNet(nn.Module):
    def __init__(self, num_classes):

        super(ResNet, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #起始层
        self.layer0_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1,padding=1,bias=False)
        self.layer0_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer0_relu = nn.ReLU(inplace=True)

        # self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #layer1.BasicBlock0
        self.layer1_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer1_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer2_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2_relu = nn.ReLU(inplace=True)
        #layer1.BasicBlock1
        self.layer3_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer3_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer4_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4_relu = nn.ReLU(inplace=True)
        # layer1.BasicBlock2
        self.layer5_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer5_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer5_relu = nn.ReLU(inplace=True)

        self.layer6_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer6_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer6_relu = nn.ReLU(inplace=True)
        #

        # layer1.BasicBlock3
        self.layer7_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer7_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7_relu = nn.ReLU(inplace=True)

        self.layer8_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer8_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer8_relu = nn.ReLU(inplace=True)

        # layer1.BasicBlock4
        self.layer9_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer9_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer9_relu = nn.ReLU(inplace=True)

        self.layer10_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10_relu = nn.ReLU(inplace=True)

        # layer1.BasicBlock5
        self.layer11_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer11_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer11_relu = nn.ReLU(inplace=True)

        self.layer12_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer12_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12_relu = nn.ReLU(inplace=True)

        # layer1.BasicBlock6
        self.layer13_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer13_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer13_relu = nn.ReLU(inplace=True)

        self.layer14_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer14_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer14_relu = nn.ReLU(inplace=True)

        # layer1.BasicBlock7
        self.layer15_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer15_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer15_relu = nn.ReLU(inplace=True)

        self.layer16_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer16_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer16_relu = nn.ReLU(inplace=True)


        # layer1.BasicBlock8
        self.layer17_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer17_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer17_relu = nn.ReLU(inplace=True)

        self.layer18_conv = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer18_norm = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer18_relu = nn.ReLU(inplace=True)


        # layer2.BasicBlock0
        self.layer19_conv = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer19_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer19_relu = nn.ReLU(inplace=True)

        self.layer20_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer20_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20_relu = nn.ReLU(inplace=True)
        # layer2.BasicBlock1
        self.layer21_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer21_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer21_relu = nn.ReLU(inplace=True)

        self.layer22_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer22_relu = nn.ReLU(inplace=True)

        # layer2.BasicBlock2
        self.layer23_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer23_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer23_relu = nn.ReLU(inplace=True)

        self.layer24_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer24_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer24_relu = nn.ReLU(inplace=True)

        # layer2.BasicBlock3
        self.layer25_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer25_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer25_relu = nn.ReLU(inplace=True)

        self.layer26_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer26_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer26_relu = nn.ReLU(inplace=True)

        # layer2.BasicBlock4
        self.layer27_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer27_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer27_relu = nn.ReLU(inplace=True)

        self.layer28_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer28_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer28_relu = nn.ReLU(inplace=True)

        # layer2.BasicBlock5
        self.layer29_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer29_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer29_relu = nn.ReLU(inplace=True)

        self.layer30_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer30_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30_relu = nn.ReLU(inplace=True)
        # layer2.BasicBlock6
        self.layer31_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer31_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer31_relu = nn.ReLU(inplace=True)

        self.layer32_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer32_relu = nn.ReLU(inplace=True)

        # layer2.BasicBlock7
        self.layer33_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer33_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer33_relu = nn.ReLU(inplace=True)

        self.layer34_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer34_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer34_relu = nn.ReLU(inplace=True)

        # layer2.BasicBlock8
        self.layer35_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer35_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer35_relu = nn.ReLU(inplace=True)

        self.layer36_conv = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer36_norm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer36_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock0
        self.layer37_conv = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer37_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer37_relu = nn.ReLU(inplace=True)

        self.layer38_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer38_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer38_relu = nn.ReLU(inplace=True)
        # layer3.BasicBlock1
        self.layer39_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer39_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer39_relu = nn.ReLU(inplace=True)

        self.layer40_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer40_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock2
        self.layer41_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer41_relu = nn.ReLU(inplace=True)

        self.layer42_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer42_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer42_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock3
        self.layer43_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer43_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer43_relu = nn.ReLU(inplace=True)

        self.layer44_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer44_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer44_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock4
        self.layer45_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer45_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer45_relu = nn.ReLU(inplace=True)

        self.layer46_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer46_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer46_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock5
        self.layer47_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer47_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer47_relu = nn.ReLU(inplace=True)

        self.layer48_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer48_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer48_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock6
        self.layer49_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer49_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer49_relu = nn.ReLU(inplace=True)

        self.layer50_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer50_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer50_relu = nn.ReLU(inplace=True)
        # layer3.BasicBlock7
        self.layer51_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer51_relu = nn.ReLU(inplace=True)

        self.layer52_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer52_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer52_relu = nn.ReLU(inplace=True)

        # layer3.BasicBlock8
        self.layer53_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer53_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer53_relu = nn.ReLU(inplace=True)

        self.layer54_conv = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer54_norm = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer54_relu = nn.ReLU(inplace=True)

        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(64, num_classes))
        ]))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):



        layer0_out = self.layer0_relu(self.layer0_norm(self.layer0_conv(x)))



        layer1_out = self.layer1_relu(self.layer1_norm(self.layer1_conv(layer0_out)))
        layer2_out = self.layer2_norm(self.layer2_conv(layer1_out))
        laye2_shortcut = layer0_out+layer2_out
        layer2_shortcut_out = self.layer2_relu(laye2_shortcut)

        layer3_out = self.layer3_relu(self.layer3_norm(self.layer3_conv(layer2_shortcut_out)))
        layer4_out = self.layer4_norm(self.layer4_conv(layer3_out))
        laye4_shortcut = layer2_shortcut_out+layer4_out
        layer4_shortcut_out = self.layer4_relu(laye4_shortcut)

        layer5_out = self.layer5_relu(self.layer5_norm(self.layer5_conv(layer4_shortcut_out)))
        layer6_out = self.layer6_norm(self.layer6_conv(layer5_out))
        laye6_shortcut = layer4_shortcut_out+layer6_out
        layer6_shortcut_out = self.layer5_relu(laye6_shortcut)


        layer7_out = self.layer7_relu(self.layer7_norm(self.layer7_conv(layer6_shortcut_out)))
        layer8_out = self.layer8_norm(self.layer8_conv(layer7_out))
        laye8_shortcut = layer6_shortcut_out+layer8_out
        layer8_shortcut_out = self.layer8_relu(laye8_shortcut)




        layer9_out = self.layer9_relu(self.layer9_norm(self.layer9_conv(layer8_shortcut_out)))
        layer10_out = self.layer10_norm(self.layer10_conv(layer9_out))
        laye10_shortcut = layer8_shortcut_out+layer10_out
        layer10_shortcut_out = self.layer10_relu(laye10_shortcut)

        layer11_out = self.layer11_relu(self.layer11_norm(self.layer11_conv(layer10_shortcut_out)))
        layer12_out = self.layer12_norm(self.layer12_conv(layer11_out))
        laye12_shortcut = layer10_shortcut_out+layer12_out
        layer12_shortcut_out = self.layer12_relu(laye12_shortcut)

        layer13_out = self.layer13_relu(self.layer13_norm(self.layer13_conv(layer12_shortcut_out)))
        layer14_out = self.layer14_norm(self.layer14_conv(layer13_out))
        laye14_shortcut = layer12_shortcut_out+layer14_out
        layer14_shortcut_out = self.layer14_relu(laye14_shortcut)

        # layer3--------------------------------------------------------------------------------------

        layer15_out = self.layer15_relu(self.layer15_norm(self.layer15_conv(layer14_shortcut_out)))
        layer16_out = self.layer16_norm(self.layer16_conv(layer15_out))
        laye16_shortcut = layer14_shortcut_out+layer16_out
        layer16_shortcut_out = self.layer16_relu(laye16_shortcut)


        layer17_out = self.layer17_relu(self.layer17_norm(self.layer17_conv(layer16_shortcut_out)))
        layer18_out = self.layer18_norm(self.layer18_conv(layer17_out))
        laye18_shortcut = layer16_shortcut_out+layer18_out
        layer18_shortcut_out = self.layer18_relu(laye18_shortcut)

        layer19_out = self.layer19_relu(self.layer19_norm(self.layer19_conv(layer18_shortcut_out)))
        layer20_out = self.layer20_norm(self.layer20_conv(layer19_out))
        layer18_shortcut_out_paded = self.short_cut_layer(layer18_shortcut_out,layer19_out.shape[1],layer20_out.shape[1])
        layer20_out+=layer18_shortcut_out_paded
        layer20_shortcut_out = self.layer20_relu(layer20_out)

        layer21_out = self.layer21_relu(self.layer21_norm(self.layer21_conv(layer20_shortcut_out)))
        layer22_out = self.layer22_norm(self.layer22_conv(layer21_out))
        laye22_shortcut = layer20_shortcut_out+layer22_out
        layer22_shortcut_out = self.layer22_relu(laye22_shortcut)

        layer23_out = self.layer23_relu(self.layer23_norm(self.layer23_conv(layer22_shortcut_out)))
        layer24_out = self.layer24_norm(self.layer24_conv(layer23_out))
        laye24_shortcut = layer22_shortcut_out+layer24_out
        layer24_shortcut_out = self.layer24_relu(laye24_shortcut)



        layer25_out = self.layer25_relu(self.layer25_norm(self.layer25_conv(layer24_shortcut_out)))
        layer26_out = self.layer26_norm(self.layer26_conv(layer25_out))
        laye26_shortcut = layer24_shortcut_out+layer26_out
        layer26_shortcut_out = self.layer26_relu(laye26_shortcut)


        layer27_out = self.layer27_relu(self.layer27_norm(self.layer27_conv(layer26_shortcut_out)))
        layer28_out = self.layer28_norm(self.layer28_conv(layer27_out))
        laye28_shortcut = layer26_shortcut_out+layer28_out
        layer28_shortcut_out = self.layer28_relu(laye28_shortcut)

        layer29_out = self.layer29_relu(self.layer29_norm(self.layer29_conv(layer28_shortcut_out)))
        layer30_out = self.layer30_norm(self.layer30_conv(layer29_out))
        laye30_shortcut = layer28_shortcut_out+layer30_out
        layer30_shortcut_out = self.layer30_relu(laye30_shortcut)




        layer31_out = self.layer31_relu(self.layer31_norm(self.layer31_conv(layer30_shortcut_out)))
        layer32_out = self.layer32_norm(self.layer32_conv(layer31_out))
        laye32_shortcut = layer30_shortcut_out+layer32_out
        layer32_shortcut_out = self.layer32_relu(laye32_shortcut)

        layer33_out = self.layer33_relu(self.layer33_norm(self.layer33_conv(layer32_shortcut_out)))
        layer34_out = self.layer34_norm(self.layer34_conv(layer33_out))
        laye34_shortcut = layer32_shortcut_out+layer34_out
        layer34_shortcut_out = self.layer34_relu(laye34_shortcut)


        layer35_out = self.layer35_relu(self.layer35_norm(self.layer35_conv(layer34_shortcut_out)))
        layer36_out = self.layer36_norm(self.layer36_conv(layer35_out))
        laye36_shortcut = layer34_shortcut_out+layer36_out
        layer36_shortcut_out = self.layer36_relu(laye36_shortcut)

        layer37_out = self.layer37_relu(self.layer37_norm(self.layer37_conv(layer36_shortcut_out)))
        layer38_out = self.layer38_norm(self.layer38_conv(layer37_out))
        layer36_shortcut_out_paded = self.short_cut_layer(layer36_shortcut_out,layer37_out.shape[1],layer38_out.shape[1])
        layer38_out += layer36_shortcut_out_paded
        layer38_shortcut_out = self.layer38_relu(layer38_out)

        layer39_out = self.layer39_relu(self.layer39_norm(self.layer39_conv(layer38_shortcut_out)))
        layer40_out = self.layer40_norm(self.layer40_conv(layer39_out))
        layer40_shortcut = layer38_shortcut_out+layer40_out
        layert40_shortcut_out = self.layer40_relu(layer40_shortcut)

        layer41_out = self.layer41_relu(self.layer41_norm(self.layer41_conv(layert40_shortcut_out)))
        layer42_out = self.layer42_norm(self.layer42_conv(layer41_out))
        laye42_shortcut = layert40_shortcut_out+layer42_out
        layer42_shortcut_out = self.layer42_relu(laye42_shortcut)

        layer43_out = self.layer43_relu(self.layer43_norm(self.layer43_conv(layer42_shortcut_out)))
        layer44_out = self.layer44_norm(self.layer44_conv(layer43_out))
        layer44_shortcut = layer42_shortcut_out+layer44_out
        layer44_shortcut_out = self.layer44_relu(layer44_shortcut)

        layer45_out = self.layer45_relu(self.layer45_norm(self.layer45_conv(layer44_shortcut_out)))
        layer46_out = self.layer46_norm(self.layer46_conv(layer45_out))
        laye46_shortcut = layer44_shortcut+layer46_out
        layer46_shortcut_out = self.layer46_relu(laye46_shortcut)

        layer47_out = self.layer47_relu(self.layer47_norm(self.layer47_conv(layer46_shortcut_out)))
        layer48_out = self.layer48_norm(self.layer48_conv(layer47_out))
        laye48_shortcut = layer46_shortcut_out+layer48_out
        layer48_shortcut_out = self.layer48_relu(laye48_shortcut)

        layer49_out = self.layer49_relu(self.layer49_norm(self.layer49_conv(layer48_shortcut_out)))
        layer50_out = self.layer50_norm(self.layer50_conv(layer49_out))
        laye50_shortcut = layer48_shortcut_out+layer50_out
        layer50_shortcut_out = self.layer50_relu(laye50_shortcut)

        layer51_out = self.layer51_relu(self.layer51_norm(self.layer51_conv(layer50_shortcut_out)))
        layer52_out = self.layer52_norm(self.layer52_conv(layer51_out))
        laye52_shortcut = layer50_shortcut_out+layer52_out
        layer52_shortcut_out = self.layer52_relu(laye52_shortcut)

        layer53_out = self.layer53_relu(self.layer53_norm(self.layer53_conv(layer52_shortcut_out)))
        layer54_out = self.layer54_norm(self.layer54_conv(layer53_out))
        laye54_shortcut = layer52_shortcut_out+layer54_out
        layer54_shortcut_out = self.layer54_relu(laye54_shortcut)

        # pool1_out= self.pool1(layer32_out)
        y = self.avg_pool(layer54_shortcut_out)
        # y = F.pad(y, pad=(0, 0, 0, 0, 64 - y.shape[1], 0), mode='constant', value=0)
        y = y.view(y.size(0), -1)

        out_fc = self.classifier(y)
        return out_fc


    def short_cut(self,input_tensor, output_tesnor):
        # input_tensor = input_tensor[:, :, ::2, ::2]
        input_channel = input_tensor.shape[1]
        output_channel = output_tesnor.shape[1]
        pad_channels = output_channel - input_channel
        input_tensor = F.pad(input_tensor[:, :, ::2, ::2], pad=(0, 0, 0, 0,pad_channels,0), mode='constant', value=0)
            # # input_tensor = F.pad(input_tensor, pad=(0, 0, 0, 0,pad_channels,0), mode='constant', value=0)
            # input_tensor = F.pad(input_tensor[:, :, ::2, ::2], (0, 0, 0, 0, pad_channels // 2, pad_channels // 2), "constant", 0)

        return input_tensor + output_tesnor

    def short_cut_layer(self,x,planes,target_planes):
        if x.shape[1]+(planes // 4)*2 == target_planes:
            return  F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)
        else:
            return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, (planes // 4)+1), "constant", 0)



def resnet_56(num_class):
    return ResNet(num_class)




if __name__ == '__main__':
    model = resnet_56(10)
    print(model)

