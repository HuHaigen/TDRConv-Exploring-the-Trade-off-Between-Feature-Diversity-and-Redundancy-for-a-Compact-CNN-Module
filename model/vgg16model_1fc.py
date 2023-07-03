import math
import torch.nn as nn
from collections import OrderedDict
from model.TDRConv import TDRConv
from model.SEConv import conv3x3_2re,SEConv_serial

def conv3x3_test(in_planes, out_planes, kernel_size=3, padding=1, bias=False, stride=(1, 1)):
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)
    stride = stride[0]
    return TDRConv(in_planes, out_planes, kernel_size=3, stride=stride,in_ratio=2, out_ratio=2,exp_times=4)

class VGG_test(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGG_test, self).__init__()
        # self.layer0_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer0_conv = conv3x3_test(3, 64, kernel_size=3, padding=1)
        self.layer0_norm = nn.BatchNorm2d(64)
        self.layer0_relu = nn.ReLU(inplace=True)
        self.layer1_conv = conv3x3_test(64, 64, kernel_size=3, padding=1)
        self.layer1_norm = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2_conv = conv3x3_test(64, 128, kernel_size=3, padding=1)
        self.layer2_norm = nn.BatchNorm2d(128)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_conv = conv3x3_test(128, 128, kernel_size=3, padding=1)
        self.layer3_norm = nn.BatchNorm2d(128)
        self.layer3_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4_conv = conv3x3_test(128, 256, kernel_size=3, padding=1)
        self.layer4_norm = nn.BatchNorm2d(256)
        self.layer4_relu = nn.ReLU(inplace=True)

        self.layer5_conv = conv3x3_test(256, 256, kernel_size=3, padding=1)
        self.layer5_norm = nn.BatchNorm2d(256)
        self.layer5_relu = nn.ReLU(inplace=True)
        self.layer6_conv = conv3x3_test(256, 256, kernel_size=3, padding=1)
        self.layer6_norm = nn.BatchNorm2d(256)
        self.layer6_relu = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer7_conv = conv3x3_test(256, 512, kernel_size=3, padding=1)
        self.layer7_norm = nn.BatchNorm2d(512)
        self.layer7_relu = nn.ReLU(inplace=True)
        self.layer8_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer8_norm = nn.BatchNorm2d(512)
        self.layer8_relu = nn.ReLU(inplace=True)

        self.layer9_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer9_norm = nn.BatchNorm2d(512)
        self.layer9_relu = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer10_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer10_norm = nn.BatchNorm2d(512)
        self.layer10_relu = nn.ReLU(inplace=True)
        self.layer11_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer11_norm = nn.BatchNorm2d(512)
        self.layer11_relu = nn.ReLU(inplace=True)
        self.layer12_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer12_norm = nn.BatchNorm2d(512)
        self.layer12_relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, num_classes))
        ]))

        if init_weights:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, origin_x):
        origin_x = self.layer0_conv(origin_x)
        origin_x = self.layer0_norm(origin_x)
        origin_x = self.layer0_relu(origin_x)
        # test = origin_x.detach()
        # count = np.where(np.corrcoef(test[0].view(64, -1).cpu().numpy()) > 0.9)[0].size
        # print(count)
        origin_x = self.layer1_conv(origin_x)
        origin_x = self.layer1_norm(origin_x)
        origin_x = self.layer1_relu(origin_x)

        origin_x = self.pool0(origin_x)
        origin_x = self.layer2_conv(origin_x)
        origin_x = self.layer2_norm(origin_x)
        origin_x = self.layer2_relu(origin_x)

        origin_x = self.layer3_conv(origin_x)
        origin_x = self.layer3_norm(origin_x)
        origin_x = self.layer3_relu(origin_x)

        origin_x = self.pool1(origin_x)
        origin_x = self.layer4_conv(origin_x)
        origin_x = self.layer4_norm(origin_x)
        origin_x = self.layer4_relu(origin_x)

        origin_x = self.layer5_conv(origin_x)
        origin_x = self.layer5_norm(origin_x)
        origin_x = self.layer5_relu(origin_x)
        origin_x = self.layer6_conv(origin_x)
        origin_x = self.layer6_norm(origin_x)
        origin_x = self.layer6_relu(origin_x)
        origin_x = self.pool2(origin_x)
        origin_x = self.layer7_conv(origin_x)
        origin_x = self.layer7_norm(origin_x)
        origin_x = self.layer7_relu(origin_x)
        origin_x = self.layer8_conv(origin_x)
        origin_x = self.layer8_norm(origin_x)
        origin_x = self.layer8_relu(origin_x)
        origin_x = self.layer9_conv(origin_x)
        origin_x = self.layer9_norm(origin_x)
        origin_x = self.layer9_relu(origin_x)
        origin_x = self.pool3(origin_x)
        origin_x = self.layer10_conv(origin_x)
        origin_x = self.layer10_norm(origin_x)
        origin_x = self.layer10_relu(origin_x)
        origin_x = self.layer11_conv(origin_x)
        origin_x = self.layer11_norm(origin_x)
        origin_x = self.layer11_relu(origin_x)
        origin_x = self.layer12_conv(origin_x)
        origin_x = self.layer12_norm(origin_x)
        origin_x = self.layer12_relu(origin_x)
        origin_x = nn.AvgPool2d(2)(origin_x)
        origin_x = origin_x.view(origin_x.size(0), -1)
        out_fc = self.classifier(origin_x)

        return out_fc



class VGG_test_returnSimilarity(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGG_test_returnSimilarity, self).__init__()
        # self.layer0_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer0_conv = conv3x3_test(3, 64, kernel_size=3, padding=1)
        self.layer0_norm = nn.BatchNorm2d(64)
        self.layer0_relu = nn.ReLU(inplace=True)
        self.layer1_conv = conv3x3_test(64, 64, kernel_size=3, padding=1)
        self.layer1_norm = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2_conv = conv3x3_test(64, 128, kernel_size=3, padding=1)
        self.layer2_norm = nn.BatchNorm2d(128)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_conv = conv3x3_test(128, 128, kernel_size=3, padding=1)
        self.layer3_norm = nn.BatchNorm2d(128)
        self.layer3_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4_conv = conv3x3_test(128, 256, kernel_size=3, padding=1)
        self.layer4_norm = nn.BatchNorm2d(256)
        self.layer4_relu = nn.ReLU(inplace=True)

        self.layer5_conv = conv3x3_test(256, 256, kernel_size=3, padding=1)
        self.layer5_norm = nn.BatchNorm2d(256)
        self.layer5_relu = nn.ReLU(inplace=True)
        self.layer6_conv = conv3x3_test(256, 256, kernel_size=3, padding=1)
        self.layer6_norm = nn.BatchNorm2d(256)
        self.layer6_relu = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer7_conv = conv3x3_test(256, 512, kernel_size=3, padding=1)
        self.layer7_norm = nn.BatchNorm2d(512)
        self.layer7_relu = nn.ReLU(inplace=True)
        self.layer8_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer8_norm = nn.BatchNorm2d(512)
        self.layer8_relu = nn.ReLU(inplace=True)

        self.layer9_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer9_norm = nn.BatchNorm2d(512)
        self.layer9_relu = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer10_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer10_norm = nn.BatchNorm2d(512)
        self.layer10_relu = nn.ReLU(inplace=True)
        self.layer11_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer11_norm = nn.BatchNorm2d(512)
        self.layer11_relu = nn.ReLU(inplace=True)
        self.layer12_conv = conv3x3_test(512, 512, kernel_size=3, padding=1)
        self.layer12_norm = nn.BatchNorm2d(512)
        self.layer12_relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, num_classes))
        ]))

        if init_weights:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, origin_x):
        nn.Module
        s_list =[]
        origin_x,s = self.layer0_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer0_norm(origin_x)
        origin_x = self.layer0_relu(origin_x)
        # test = origin_x.detach()
        # count = np.where(np.corrcoef(test[0].view(64, -1).cpu().numpy()) > 0.9)[0].size
        # print(count)
        origin_x,s = self.layer1_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer1_norm(origin_x)
        origin_x = self.layer1_relu(origin_x)

        origin_x = self.pool0(origin_x)
        origin_x,s = self.layer2_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer2_norm(origin_x)
        origin_x = self.layer2_relu(origin_x)

        origin_x,s = self.layer3_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer3_norm(origin_x)
        origin_x = self.layer3_relu(origin_x)

        origin_x = self.pool1(origin_x)
        origin_x,s = self.layer4_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer4_norm(origin_x)
        origin_x = self.layer4_relu(origin_x)

        origin_x,s = self.layer5_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer5_norm(origin_x)
        origin_x = self.layer5_relu(origin_x)
        
        origin_x,s = self.layer6_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer6_norm(origin_x)
        origin_x = self.layer6_relu(origin_x)
        origin_x = self.pool2(origin_x)
        
        origin_x,s = self.layer7_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer7_norm(origin_x)
        origin_x = self.layer7_relu(origin_x)
        
        origin_x,s = self.layer8_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer8_norm(origin_x)
        origin_x = self.layer8_relu(origin_x)
        
        origin_x,s = self.layer9_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer9_norm(origin_x)
        origin_x = self.layer9_relu(origin_x)
        origin_x = self.pool3(origin_x)
        
        origin_x,s = self.layer10_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer10_norm(origin_x)
        origin_x = self.layer10_relu(origin_x)
        
        origin_x,s = self.layer11_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer11_norm(origin_x)
        origin_x = self.layer11_relu(origin_x)
        
        origin_x,s = self.layer12_conv(origin_x)
        s_list.append(s)
        origin_x = self.layer12_norm(origin_x)
        origin_x = self.layer12_relu(origin_x)
        origin_x = nn.AvgPool2d(2)(origin_x)
        origin_x = origin_x.view(origin_x.size(0), -1)
        out_fc = self.classifier(origin_x)

        return out_fc,s_list

class VGG(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer0_norm = nn.BatchNorm2d(64)
        self.layer0_relu = nn.ReLU(inplace=True)
        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.layer1_norm = nn.BatchNorm2d(64)
        self.layer1_relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer2_norm = nn.BatchNorm2d(128)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer3_norm = nn.BatchNorm2d(128)
        self.layer3_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer4_norm = nn.BatchNorm2d(256)
        self.layer4_relu = nn.ReLU(inplace=True)

        self.layer5_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer5_norm = nn.BatchNorm2d(256)
        self.layer5_relu = nn.ReLU(inplace=True)
        self.layer6_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer6_norm = nn.BatchNorm2d(256)
        self.layer6_relu = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer7_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.layer7_norm = nn.BatchNorm2d(512)
        self.layer7_relu = nn.ReLU(inplace=True)
        self.layer8_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer8_norm = nn.BatchNorm2d(512)
        self.layer8_relu = nn.ReLU(inplace=True)

        self.layer9_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer9_norm = nn.BatchNorm2d(512)
        self.layer9_relu = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer10_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer10_norm = nn.BatchNorm2d(512)
        self.layer10_relu = nn.ReLU(inplace=True)
        self.layer11_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer11_norm = nn.BatchNorm2d(512)
        self.layer11_relu = nn.ReLU(inplace=True)
        self.layer12_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer12_norm = nn.BatchNorm2d(512)
        self.layer12_relu = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, num_classes))
        ]))

        if init_weights:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, origin_x):
        origin_x = self.layer0_conv(origin_x)
        origin_x = self.layer0_norm(origin_x)
        origin_x = self.layer0_relu(origin_x)
        # test = origin_x.detach()
        # count = np.where(np.corrcoef(test[0].view(64, -1).cpu().numpy()) > 0.9)[0].size
        # print(count)
        origin_x = self.layer1_conv(origin_x)
        origin_x = self.layer1_norm(origin_x)
        origin_x = self.layer1_relu(origin_x)

        origin_x = self.pool0(origin_x)
        origin_x = self.layer2_conv(origin_x)
        origin_x = self.layer2_norm(origin_x)
        origin_x = self.layer2_relu(origin_x)

        origin_x = self.layer3_conv(origin_x)
        origin_x = self.layer3_norm(origin_x)
        origin_x = self.layer3_relu(origin_x)

        origin_x = self.pool1(origin_x)
        origin_x = self.layer4_conv(origin_x)
        origin_x = self.layer4_norm(origin_x)
        origin_x = self.layer4_relu(origin_x)

        origin_x = self.layer5_conv(origin_x)
        origin_x = self.layer5_norm(origin_x)
        origin_x = self.layer5_relu(origin_x)
        origin_x = self.layer6_conv(origin_x)
        origin_x = self.layer6_norm(origin_x)
        origin_x = self.layer6_relu(origin_x)
        origin_x = self.pool2(origin_x)
        origin_x = self.layer7_conv(origin_x)
        origin_x = self.layer7_norm(origin_x)
        origin_x = self.layer7_relu(origin_x)
        origin_x = self.layer8_conv(origin_x)
        origin_x = self.layer8_norm(origin_x)
        origin_x = self.layer8_relu(origin_x)
        origin_x = self.layer9_conv(origin_x)
        origin_x = self.layer9_norm(origin_x)
        origin_x = self.layer9_relu(origin_x)
        origin_x = self.pool3(origin_x)
        origin_x = self.layer10_conv(origin_x)
        origin_x = self.layer10_norm(origin_x)
        origin_x = self.layer10_relu(origin_x)
        origin_x = self.layer11_conv(origin_x)
        origin_x = self.layer11_norm(origin_x)
        origin_x = self.layer11_relu(origin_x)
        origin_x = self.layer12_conv(origin_x)
        origin_x = self.layer12_norm(origin_x)
        origin_x = self.layer12_relu(origin_x)
        origin_x = nn.AvgPool2d(2)(origin_x)
        origin_x = origin_x.view(origin_x.size(0), -1)
        out_fc = self.classifier(origin_x)

        return out_fc


def vgg_16_1fc(num_class):
    return VGG(num_class)

def vgg_16_1fc_tdrc(num_class):
    return VGG_test(num_class)
