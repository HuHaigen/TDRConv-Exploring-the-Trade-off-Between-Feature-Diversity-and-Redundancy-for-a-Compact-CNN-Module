from torch import nn
import numpy as np
from collections import Counter, OrderedDict
import torch
from pathlib import Path
import re
import torchvision
from utils.imagenet import Data
from torchvision import transforms
import copy
from utils.likelyhood_utils import get_model


def count_params(model, input_size=224):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))
 
    # 计算模型的计算量
    calc_flops(model, input_size)
 
    # 计算模型的参数总量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
 
    print('The network has {} params.'.format(params))
 
 
# 计算模型的计算量
def calc_flops(model, input_size=224,use_gpu = True):
 
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_conv.append(flops)
 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
 
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
 
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
 
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_pooling.append(flops)
 
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
 
    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    if '0.4.' in torch.__version__:
        # if assets.USE_GPU:
        if use_gpu:
            print('assets.USE_GPU')
            input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
        else:
            input = torch.FloatTensor(torch.rand(2, 3, input_size, input_size))
    else:
        input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
        # input = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)
    _ = model(input)
 
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
 
    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6 / 2))






def prune_model(model,compress_rate,network):
    #先计算有多少个卷积层
    cov_count = 53
    count_dict = get_filter_idx(model['counter_dict'], compress_rate)
    net_pruned = ''
    if network == 'vgg_16' or network == 'vgg_16_1fc':
        for idx in range(cov_count - 1):
            layer_idx = str(idx)
            net_pruned = prune_layers(count_dict, layer_idx, model['net'])

    elif network == 'resnet_34':
        for idx in range(cov_count - 1):
            layer_idx = str(idx)
            net_pruned = prune_resnet_layers(count_dict, layer_idx, model['net'])
        reconstruct_downsample(net_pruned)
    elif network == 'resnet_50':
        skip_list = [4,14,27,46]
        for idx in range(cov_count):
            if idx in skip_list:
                continue
            layer_idx = str(idx)
            net_pruned = prune_resnet_layers(count_dict, layer_idx, model['net'])
        reconstruct_50_downsample(net_pruned)
    elif network == 'resnet_56':
        for idx in range(cov_count):
            layer_idx = str(idx)
            net_pruned = prune_resnet_layers(count_dict, layer_idx, model['net'])
    elif network == 'googlenet':
        for idx in range(cov_count):
            layer_idx = str(idx)
            net_pruned = prune_googlenet_layers(count_dict, layer_idx, model['net'])
            reconstruct_googlenet_branch11(net_pruned)
            reconstruct_googlenet_branch_componets(net_pruned)
    elif network == 'densenet_121':
        for idx in range(1, cov_count - 1):
            layer_idx = str(idx)
            net_pruned = prune_densenet_141_layers(count_dict, layer_idx, model['net'])
        reconstruct_transition(net_pruned)
        reconstruct_classifier(net_pruned)
    elif network == 'densenet_40':
        for idx in range(1, cov_count - 1):
            layer_idx = str(idx)
            net_pruned = prune_densenet_40_layers(count_dict, layer_idx, model['net'])
        reconstruct_40_classifier(net_pruned)
    elif network == 'densenet_50':
        for idx in range(1, cov_count - 1):
            layer_idx = str(idx)
            net_pruned = prune_densenet_40_layers(count_dict, layer_idx, model['net'])
        reconstruct_50_classifier(net_pruned)
    else:
        net_pruned = '';
    model['net'] = net_pruned
    return model


def prune_layers(count_dict, layer_idx, model):
    layer_name = 'layer' + layer_idx
    layer_conv_name = layer_name + '_conv'  # 卷积层名称
    filter_list = count_dict[layer_conv_name]
    filter_in_channels = model._modules[layer_conv_name].in_channels  # 输入通道
    filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
    filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[layer_conv_name].stride[0]
    filter_padding = model._modules[layer_conv_name].padding[0]
    total_filter_number = model.state_dict()['layer' + layer_idx + '_conv.weight'].shape[0]
    total_filter_list = list(range(total_filter_number))
    filter_idx_keep = list(set(total_filter_list).difference(set(filter_list)))
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep = model.state_dict()[layer_conv_name + '.weight'][filter_idx_keep]

    conv_bias_keep = model.state_dict()[layer_conv_name + '.bias'][filter_idx_keep]

    norm_weight_keep = model.state_dict()[layer_name + '_norm.weight'][filter_idx_keep]
    norm_bias_keep = model.state_dict()[layer_name + '_norm.bias'][filter_idx_keep]
    norm_mean_keep = model.state_dict()[layer_name + '_norm.running_mean'][filter_idx_keep]
    norm_var_keep = model.state_dict()[layer_name + '_norm.running_var'][filter_idx_keep]
    # 重建网络结构,
    model._modules['layer' + layer_idx + '_conv'] = nn.Conv2d(filter_in_channels, filter_idx_keep.__len__(),
                                                              kernel_size=filter_size, stride=stride_size,
                                                              padding=filter_padding)
    model._modules['layer' + layer_idx + '_norm'] = nn.BatchNorm2d(filter_idx_keep.__len__())
    model.state_dict()[layer_conv_name + '.weight'].copy_(conv_weight_keep)
    model.state_dict()[layer_conv_name + '.bias'].copy_(conv_bias_keep)
    model.state_dict()[layer_name + '_norm.weight'].copy_(norm_weight_keep)
    model.state_dict()[layer_name + '_norm.bias'].copy_(norm_bias_keep)
    model.state_dict()[layer_name + '_norm.running_mean'].copy_(norm_mean_keep)
    model.state_dict()[layer_name + '_norm.running_var'].copy_(norm_var_keep)
    last_conv_idx = get_last_convidx(model)
    if int(layer_idx) == last_conv_idx:
        model = prune_fc_layer(model, filter_idx_keep)
        return model

    #处理下一层的
    next_layer_index = int(layer_idx)+ 1
    next_layer_name = 'layer' + str(next_layer_index)
    next_layer_name_conv = next_layer_name+'_conv'

    filter_in_channels = model._modules[next_layer_name_conv].in_channels  # 输入通道
    filter_out_channels = model._modules[next_layer_name_conv].out_channels  # 输出通道
    filter_size = model._modules[next_layer_name_conv].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[next_layer_name_conv].stride[0]  # stride大小
    filter_padding = model._modules[next_layer_name_conv].padding[0]
    total_filter_thickness = model.state_dict()[next_layer_name_conv+'.weight'].shape[1]
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep_thickness = model.state_dict()[next_layer_name_conv + '.weight'][:,filter_idx_keep,:,:]
    model._modules[next_layer_name_conv] = nn.Conv2d(len(filter_idx_keep), filter_out_channels,
                                                              kernel_size=filter_size, padding=filter_padding,stride=stride_size)
    model.state_dict()[next_layer_name_conv + '.weight'].copy_(conv_weight_keep_thickness)

    return model

def prune_googlenet_layers(count_dict, layer_idx, model):
    layer_name = 'layer' + layer_idx
    layer_conv_name = layer_name + '_conv'  # 卷积层名称
    filter_list = count_dict[layer_conv_name]
    filter_in_channels = model._modules[layer_conv_name].in_channels  # 输入通道
    filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
    filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[layer_conv_name].stride[0]
    filter_padding = model._modules[layer_conv_name].padding[0]
    total_filter_number = model.state_dict()['layer' + layer_idx + '_conv.weight'].shape[0]
    total_filter_list = list(range(total_filter_number))
    filter_idx_keep = list(set(total_filter_list).difference(set(filter_list)))
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep = model.state_dict()[layer_conv_name + '.weight'][filter_idx_keep]

    norm_weight_keep = model.state_dict()[layer_name + '_norm.weight'][filter_idx_keep]
    norm_bias_keep = model.state_dict()[layer_name + '_norm.bias'][filter_idx_keep]
    norm_mean_keep = model.state_dict()[layer_name + '_norm.running_mean'][filter_idx_keep]
    norm_var_keep = model.state_dict()[layer_name + '_norm.running_var'][filter_idx_keep]

    #重建网络结构,
    model._modules['layer' + layer_idx + '_conv'] = nn.Conv2d(filter_in_channels, filter_idx_keep.__len__(),
                                                              kernel_size=filter_size, stride=stride_size,padding=filter_padding)
    model._modules['layer' + layer_idx + '_norm'] = nn.BatchNorm2d(filter_idx_keep.__len__())
    model.state_dict()[layer_conv_name + '.weight'].copy_(conv_weight_keep)
    model.state_dict()[layer_name + '_norm.weight'].copy_(norm_weight_keep)
    model.state_dict()[layer_name + '_norm.bias'].copy_(norm_bias_keep)
    model.state_dict()[layer_name + '_norm.running_mean'].copy_(norm_mean_keep)
    model.state_dict()[layer_name + '_norm.running_var'].copy_(norm_var_keep)
    last_conv_idx = get_last_convidx(model)
    if int(layer_idx) == last_conv_idx:
        model = prune_fc_layer(model, filter_idx_keep)
        return model

    #处理下一层的
    next_layer_index = int(layer_idx)+ 1
    next_layer_name = 'layer' + str(next_layer_index)
    next_layer_name_conv = next_layer_name+'_conv'

    filter_in_channels = model._modules[next_layer_name_conv].in_channels  # 输入通道
    filter_out_channels = model._modules[next_layer_name_conv].out_channels  # 输出通道
    filter_size = model._modules[next_layer_name_conv].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[next_layer_name_conv].stride[0]#stride大小
    filter_padding = model._modules[next_layer_name_conv].padding[0]
    total_filter_thickness = model.state_dict()[next_layer_name_conv+'.weight'].shape[1]
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep_thickness = model.state_dict()[next_layer_name_conv + '.weight'][:,filter_idx_keep,:,:]
    model._modules[next_layer_name_conv] = nn.Conv2d(len(filter_idx_keep), filter_out_channels,
                                                              kernel_size=filter_size,stride=stride_size,padding=filter_padding)
    model.state_dict()[next_layer_name_conv + '.weight'].copy_(conv_weight_keep_thickness)

    return model

def prune_resnet_layers(count_dict, layer_idx, model):
    layer_name = 'layer' + layer_idx
    layer_conv_name = layer_name + '_conv'  # 卷积层名称
    filter_list = count_dict[layer_conv_name]
    filter_in_channels = model._modules[layer_conv_name].in_channels  # 输入通道
    filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
    filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[layer_conv_name].stride[0]
    filter_padding = model._modules[layer_conv_name].padding[0]
    total_filter_number = model.state_dict()['layer' + layer_idx + '_conv.weight'].shape[0]
    total_filter_list = list(range(total_filter_number))
    filter_idx_keep = list(set(total_filter_list).difference(set(filter_list)))
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep = model.state_dict()[layer_conv_name + '.weight'][filter_idx_keep]

    norm_weight_keep = model.state_dict()[layer_name + '_norm.weight'][filter_idx_keep]
    norm_bias_keep = model.state_dict()[layer_name + '_norm.bias'][filter_idx_keep]
    norm_mean_keep = model.state_dict()[layer_name + '_norm.running_mean'][filter_idx_keep]
    norm_var_keep = model.state_dict()[layer_name + '_norm.running_var'][filter_idx_keep]

    #重建网络结构,
    model._modules['layer' + layer_idx + '_conv'] = nn.Conv2d(filter_in_channels, filter_idx_keep.__len__(),
                                                              kernel_size=filter_size, stride=stride_size,padding=filter_padding)
    model._modules['layer' + layer_idx + '_norm'] = nn.BatchNorm2d(filter_idx_keep.__len__())
    model.state_dict()[layer_conv_name + '.weight'].copy_(conv_weight_keep)
    model.state_dict()[layer_name + '_norm.weight'].copy_(norm_weight_keep)
    model.state_dict()[layer_name + '_norm.bias'].copy_(norm_bias_keep)
    model.state_dict()[layer_name + '_norm.running_mean'].copy_(norm_mean_keep)
    model.state_dict()[layer_name + '_norm.running_var'].copy_(norm_var_keep)
    last_conv_idx = get_last_convidx(model)
    if int(layer_idx) == last_conv_idx:
        model = prune_fc_layer(model, filter_idx_keep)
        return model

    #处理下一层的
    next_layer_index = int(layer_idx)+ 1
    next_layer_name = 'layer' + str(next_layer_index)
    next_layer_name_conv = next_layer_name+'_conv'

    filter_in_channels = model._modules[next_layer_name_conv].in_channels  # 输入通道
    filter_out_channels = model._modules[next_layer_name_conv].out_channels  # 输出通道
    filter_size = model._modules[next_layer_name_conv].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[next_layer_name_conv].stride[0]#stride大小
    filter_padding = model._modules[next_layer_name_conv].padding[0]
    total_filter_thickness = model.state_dict()[next_layer_name_conv+'.weight'].shape[1]
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    # conv_weight_keep_thickness = model.state_dict()[next_layer_name_conv + '.weight'][:,filter_idx_keep,:,:]
    model._modules[next_layer_name_conv] = nn.Conv2d(len(filter_idx_keep), filter_out_channels,
                                                              kernel_size=filter_size,stride=stride_size,padding=filter_padding)
    # model.state_dict()[next_layer_name_conv + '.weight'].copy_(conv_weight_keep_thickness)

    return model

def prune_densenet_141_layers(count_dict, layer_idx, model):
    layer_name = 'layer' + layer_idx
    last_layer_name_conv2 = 'layer' + str(int(layer_idx)-1) +'_conv2'
    l_last_layer_name_conv2 = 'layer' + str(int(layer_idx) - 2) + '_conv2'
    layer_name_norm1 = 'layer' + layer_idx+ '_norm1'
    layer_conv1_name = layer_name + '_conv1'
    layer_conv2_name = layer_name + '_conv2'  # 卷积层名称
    filter_list = count_dict[layer_conv2_name]
    filter_in_channels = model._modules[layer_conv2_name].in_channels  # 输入通道
    filter_out_channels = model._modules[layer_conv2_name].out_channels  # 输出通道
    filter_size = model._modules[layer_conv2_name].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[layer_conv2_name].stride[0]
    filter_padding = model._modules[layer_conv2_name].padding[0]
    total_filter_number = model.state_dict()['layer' + layer_idx + '_conv2.weight'].shape[0]
    total_filter_list = list(range(total_filter_number))
    filter_idx_keep = list(set(total_filter_list).difference(set(filter_list)))
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep = model.state_dict()[layer_conv2_name + '.weight'][filter_idx_keep]


    #重建网络结构,
    model._modules['layer' + layer_idx + '_conv2'] = nn.Conv2d(filter_in_channels, filter_idx_keep.__len__(),
                                                              kernel_size=filter_size, stride=stride_size,padding=filter_padding)
    model._modules['layer' + layer_idx + '_norm2'] = nn.BatchNorm2d(filter_in_channels)
    model.state_dict()[layer_conv2_name + '.weight'].copy_(conv_weight_keep)

    last_conv_idx = get_last_convidx(model)
    if int(layer_idx) == last_conv_idx:
        model = prune_fc_layer(model, filter_idx_keep)
        return model

    #处理下一层的conv1
    norm_weight_keep = model.state_dict()[layer_name + '_norm2.weight'][filter_idx_keep]
    norm_bias_keep = model.state_dict()[layer_name + '_norm2.bias'][filter_idx_keep]
    norm_mean_keep = model.state_dict()[layer_name + '_norm2.running_mean'][filter_idx_keep]
    norm_var_keep = model.state_dict()[layer_name + '_norm2.running_var'][filter_idx_keep]

    next_layer_index = int(layer_idx)+ 1
    next_layer_name = 'layer' + str(next_layer_index)
    next_layer_name_conv = next_layer_name+'_conv1'
    next_layer_name_norm = next_layer_name + '_norm1'
    last_layer_conv2_out_channels = model._modules[last_layer_name_conv2].out_channels
    last_layer_name_norm1_channels = model._modules[layer_name_norm1].num_features
    filter_in_channels = model._modules[next_layer_name_conv].in_channels  # 输入通道
    filter_out_channels = model._modules[next_layer_name_conv].out_channels  # 输出通道
    filter_size = model._modules[next_layer_name_conv].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[next_layer_name_conv].stride[0]#stride大小
    filter_padding = model._modules[next_layer_name_conv].padding[0]
    total_filter_thickness = model.state_dict()[next_layer_name_conv+'.weight'].shape[1]
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep_thickness = model.state_dict()[next_layer_name_conv + '.weight'][:,filter_idx_keep,:,:]
    model._modules[next_layer_name_norm] = nn.BatchNorm2d(len(filter_idx_keep)+last_layer_name_norm1_channels)
    model._modules[next_layer_name_conv] = nn.Conv2d(len(filter_idx_keep)+last_layer_name_norm1_channels, filter_out_channels,
                                                              kernel_size=filter_size,stride=stride_size,padding=filter_padding)
    # model.state_dict()[next_layer_name_norm + '.weight'].copy_(norm_weight_keep)
    # model.state_dict()[next_layer_name_norm + '.bias'].copy_(norm_bias_keep)
    # model.state_dict()[next_layer_name_norm + '.running_mean'].copy_(norm_mean_keep)
    # model.state_dict()[next_layer_name_norm + '.running_var'].copy_(norm_var_keep)

    # model.state_dict()[next_layer_name_conv + '.weight'].copy_(conv_weight_keep_thickness)

    return model

def prune_densenet_40_layers(count_dict, layer_idx, model):
    layer_name = 'layer' + layer_idx
    layer_conv_name = 'layer' + layer_idx +'_conv'
    last_layer_conv = 'layer' + str(int(layer_idx)-1) +'_conv'
    layer_name_norm = 'layer' + layer_idx+ '_norm'
    last_layer_norm = 'layer' + str(int(layer_idx) - 1) + '_norm'
    filter_list = count_dict[layer_conv_name]
    filter_in_channels = model._modules[layer_conv_name].in_channels  # 输入通道
    filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
    filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[layer_conv_name].stride[0]
    filter_padding = model._modules[layer_conv_name].padding[0]


    total_filter_number = model.state_dict()[layer_conv_name + '.weight'].shape[0]
    total_filter_list = list(range(total_filter_number))
    filter_idx_keep = list(set(total_filter_list).difference(set(filter_list)))
    # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    conv_weight_keep = model.state_dict()[layer_conv_name + '.weight'][filter_idx_keep]


    #重建网络结构,
    model._modules['layer' + layer_idx + '_conv'] = nn.Conv2d(filter_in_channels, filter_idx_keep.__len__(),
                                                              kernel_size=filter_size, stride=stride_size,padding=filter_padding)
    model._modules['layer' + layer_idx + '_norm'] = nn.BatchNorm2d(filter_in_channels)
    model.state_dict()[layer_conv_name + '.weight'].copy_(conv_weight_keep)

    last_conv_idx = get_last_convidx(model)
    if int(layer_idx) == last_conv_idx:
        model = prune_fc_layer(model, filter_idx_keep)
        return model

    #处理下一层的conv1
    # norm_weight_keep = model.state_dict()[layer_name + '_norm.weight'][filter_idx_keep]
    # norm_bias_keep = model.state_dict()[layer_name + '_norm.bias'][filter_idx_keep]
    # norm_mean_keep = model.state_dict()[layer_name + '_norm.running_mean'][filter_idx_keep]
    # norm_var_keep = model.state_dict()[layer_name + '_norm.running_var'][filter_idx_keep]

    next_layer_index = int(layer_idx)+ 1
    next_layer_name = 'layer' + str(next_layer_index)
    next_layer_name_conv = next_layer_name+'_conv'
    next_layer_name_norm = next_layer_name + '_norm'
    layer_conv_out_channels = model._modules[layer_conv_name].out_channels
    #13层，26层不需要进行cat操作所有不用加当前层norm的通道
    layer_name_norm_channels = 0 if layer_idx == '13'  or layer_idx =='26' else model._modules[layer_name_norm].num_features

    filter_in_channels = model._modules[next_layer_name_conv].in_channels  # 输入通道
    filter_out_channels = model._modules[next_layer_name_conv].out_channels  # 输出通道
    filter_size = model._modules[next_layer_name_conv].kernel_size[0]  # 卷积核大小
    stride_size = model._modules[next_layer_name_conv].stride[0]#stride大小
    filter_padding = model._modules[next_layer_name_conv].padding[0]
    total_filter_thickness = model.state_dict()[next_layer_name_conv+'.weight'].shape[1]

    conv_weight_keep_thickness = model.state_dict()[next_layer_name_conv + '.weight'][:,filter_idx_keep,:,:]
    model._modules[next_layer_name_norm] = nn.BatchNorm2d(len(filter_idx_keep)+layer_name_norm_channels)
    model._modules[next_layer_name_conv] = nn.Conv2d(len(filter_idx_keep)+layer_name_norm_channels, filter_out_channels,
                                                              kernel_size=filter_size,stride=stride_size,padding=filter_padding)
    # model.state_dict()[next_layer_name_norm + '.weight'].copy_(norm_weight_keep)
    # model.state_dict()[next_layer_name_norm + '.bias'].copy_(norm_bias_keep)
    # model.state_dict()[next_layer_name_norm + '.running_mean'].copy_(norm_mean_keep)
    # model.state_dict()[next_layer_name_norm + '.running_var'].copy_(norm_var_keep)

    # model.state_dict()[next_layer_name_conv + '.weight'].copy_(conv_weight_keep_thickness)

    return model

def reconstruct_downsample(model):
# # layer2.BasicBlock0.downsample
    model._modules['downsample_layer6_residual'] = nn.Conv2d(model._modules['layer6_conv'].out_channels, model._modules['layer8_conv'].in_channels, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
    model._modules['downsample_layer6_residual_norm'] = nn.BatchNorm2d(model._modules['layer8_conv'].in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()

def reconstruct_50_downsample(model):
# # layer1.ResBottleneck0.downsample
    model._modules['layer4_conv'] = nn.Conv2d(model._modules['layer1_norm'].num_features,model._modules['layer3_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
    model._modules['layer4_norm'] = nn.BatchNorm2d(model._modules['layer3_conv'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['layer5_conv'] = nn.Conv2d(model._modules['layer4_norm'].num_features,model._modules['layer5_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
# # layer2.ResBottleneck0.downsample
    model._modules['layer14_conv'] = nn.Conv2d(model._modules['layer10_norm'].num_features, model._modules['layer13_conv'].out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
    model._modules['layer14_norm'] = nn.BatchNorm2d(model._modules['layer13_conv'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['layer15_conv'] = nn.Conv2d(model._modules['layer14_norm'].num_features, model._modules['layer15_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
# # layer3.ResBottleneck0.downsample
    model._modules['layer27_conv'] = nn.Conv2d(model._modules['layer23_norm'].num_features, model._modules['layer26_conv'].out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
    model._modules['layer27_norm'] = nn.BatchNorm2d(model._modules['layer26_conv'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['layer28_conv'] = nn.Conv2d(model._modules['layer27_norm'].num_features, model._modules['layer28_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
# # layer4.ResBottleneck0.downsample
    model._modules['layer46_conv'] = nn.Conv2d(model._modules['layer42_norm'].num_features, model._modules['layer45_conv'].out_channels, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
    model._modules['layer46_norm'] = nn.BatchNorm2d(model._modules['layer45_conv'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['layer47_conv'] = nn.Conv2d(model._modules['layer46_norm'].num_features, model._modules['layer47_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
def reconstruct_transition(model):
# # transition1
    model._modules['transition1_norm'] = nn.BatchNorm2d(model._modules['layer6_norm1'].num_features+model._modules['layer6_conv2'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['transition1_conv1'] = nn.Conv2d(model._modules['layer6_norm1'].num_features+model._modules['layer6_conv2'].out_channels, model._modules['layer6_norm1'].num_features+model._modules['layer6_conv2'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
# # transition2
    model._modules['transition2_norm'] = nn.BatchNorm2d(model._modules['layer18_norm1'].num_features+model._modules['layer18_conv2'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['transition2_conv1'] = nn.Conv2d(model._modules['layer18_norm1'].num_features+model._modules['layer18_conv2'].out_channels, model._modules['layer18_norm1'].num_features+model._modules['layer18_conv2'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()

# # transition3
    model._modules['transition3_norm'] = nn.BatchNorm2d(model._modules['layer42_norm1'].num_features+model._modules['layer42_conv2'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['transition3_conv1'] = nn.Conv2d(model._modules['layer42_norm1'].num_features+model._modules['layer42_conv2'].out_channels, model._modules['layer42_norm1'].num_features+model._modules['layer42_conv2'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()

def reconstruct_40_transition(model):
# # transition1
    model._modules['transition1_norm'] = nn.BatchNorm2d(model._modules['layer11_norm'].num_features+model._modules['layer11_conv'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['transition1_conv'] = nn.Conv2d(model._modules['layer11_norm'].num_features+model._modules['layer11_conv'].out_channels, model._modules['layer11_norm'].num_features+model._modules['layer11_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
# # transition2
    model._modules['transition2_norm'] = nn.BatchNorm2d(model._modules['layer23_norm'].num_features+model._modules['layer23_conv'].out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
    model._modules['transition2_conv'] = nn.Conv2d(model._modules['layer23_norm'].num_features+model._modules['layer23_conv'].out_channels, model._modules['layer23_norm'].num_features+model._modules['layer23_conv'].out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()


def reconstruct_classifier(model):

    num_classes = model._modules['classifier']._modules['linear1'].out_features
    model._modules['classifier'] =  nn.Sequential(OrderedDict([
        ('linear1',nn.Linear(model._modules['layer58_norm1'].num_features+model._modules['layer58_conv2'].out_channels, num_classes))]))


def reconstruct_56_classifier(model):

    num_classes = model._modules['classifier']._modules['linear1'].out_features
    model._modules['classifier'] =  nn.Sequential(OrderedDict([
        ('linear1',nn.Linear(model._modules['layer54_norm'].num_features, num_classes))]))
def reconstruct_40_classifier(model):
    model._modules['layer39_norm'] = nn.BatchNorm2d(model._modules['layer38_norm'].num_features+model._modules['layer38_conv'].out_channels)

    num_classes = model._modules['classifier']._modules['linear1'].out_features
    model._modules['classifier'] =  nn.Sequential(OrderedDict([
        ('linear1',nn.Linear(model._modules['layer38_norm'].num_features+model._modules['layer38_conv'].out_channels, num_classes))]))
def reconstruct_googlenet_branch11(model):
    for i in range(8,58,7):
        layer_conv_name = 'layer'+str(i)+'_conv'
        layer_norm_name = 'layer' + str(i) + '_norm'
        onetone_norm_name = 'layer'+str(i-7)+'_norm'
        threetthree_norm_name = 'layer' + str(i - 5) + '_norm'
        fivetfive_norm_name = 'layer' + str(i - 2) + '_norm'
        pool_norm_name = 'layer' + str(i - 1) + '_norm'
        norm_channel = model._modules[onetone_norm_name].num_features+model._modules[threetthree_norm_name].num_features+model._modules[fivetfive_norm_name].num_features+model._modules[pool_norm_name].num_features
        filter_in_channels = model._modules[layer_conv_name].in_channels  # 输入通道
        filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
        filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
        stride_size = model._modules[layer_conv_name].stride[0]
        filter_padding = model._modules[layer_conv_name].padding[0]
        # 重建网络结构,
        model._modules[layer_conv_name] = nn.Conv2d(norm_channel, filter_out_channels,
                                                                  kernel_size=filter_size, stride=stride_size,
                                                                  padding=filter_padding)

def reconstruct_googlenet_branch_componets(model):
    filter_list = [1,2,4,7]
    for i in filter_list:
        layer_conv_name = 'layer' + str(i) + '_conv'
        layer_norm_name = 'layer' + str(i) + '_norm'
        norm_channel = model._modules['layer0_norm'].num_features
        filter_in_channels = model._modules[layer_conv_name].in_channels  # 输入通道
        filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
        filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
        stride_size = model._modules[layer_conv_name].stride[0]
        filter_padding = model._modules[layer_conv_name].padding[0]
        model._modules[layer_conv_name] = nn.Conv2d(norm_channel, filter_out_channels,
                                                                  kernel_size=filter_size, stride=stride_size,
                                                                  padding=filter_padding)

    for i in range(8,64,7):
        layer_index_list = [i+1,i+2,i+3,i+6]
        layer_conv_name = 'layer'+str(i)+'_conv'
        layer_norm_name = 'layer' + str(i) + '_norm'

        norm_channel = model._modules[layer_norm_name].num_features
        for j in layer_index_list:
            layer_conv_name = 'layer' + str(i) + '_conv'
            layer_norm_name = 'layer' + str(i) + '_norm'
            filter_out_channels = model._modules[layer_conv_name].out_channels  # 输出通道
            filter_size = model._modules[layer_conv_name].kernel_size[0]  # 卷积核大小
            stride_size = model._modules[layer_conv_name].stride[0]
            filter_padding = model._modules[layer_conv_name].padding[0]

            model._modules[layer_conv_name] = nn.Conv2d(norm_channel, filter_out_channels,
                                                                  kernel_size=filter_size, stride=stride_size,
                                                                  padding=filter_padding)
            # model._modules[layer_norm_name] = nn.BatchNorm2d(norm_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
def reconstruct_50_classifier(model):

    num_classes = model._modules['classifier']._modules['linear1'].out_features
    model._modules['classifier'] =  nn.Sequential(OrderedDict([
        ('linear1',nn.Linear(model._modules['layer50_norm'].num_features+model._modules['layer50_conv'].out_channels, num_classes))]))
def prune_fc_layer(model,filter_idx_keep):
    keep_length = filter_idx_keep.__len__()
    out_features = model._modules['classifier']._modules['linear1'].out_features
    weight_keep = model._modules['classifier']._modules['linear1'].weight[:, filter_idx_keep]
    model._modules['classifier']._modules['linear1'] = nn.Linear(keep_length, out_features)
    model.state_dict()['classifier.linear1.weight'].copy_(weight_keep)
    return model

def get_filter_idx(count_dict, compress_rate):
    result_dict ={}
    for k,v in count_dict.items():
        filter_total_number = len(v)
        keys = list(map(lambda x :x[0], v.most_common(int((1- compress_rate)*filter_total_number))))
        result_dict[k] = keys
    return result_dict

def get_last_convidx(net):
    idx = -1
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            idx+=1
    return idx

def get_loaders(dataset,data_dir,train_batch_size, eval_batch_size,network):

    print('==> Preparing data..')
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False, drop_last=True)
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=False,
                                                 transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False, drop_last=True)
    elif dataset == 'imagenet':
        data_tmp = Data(data_dir,train_batch_size, eval_batch_size)
        trainloader = data_tmp.loader_train
        testloader = data_tmp.loader_test
        
    elif dataset == 'mini-imagenet':
        data_tmp = Data(data_dir,train_batch_size, eval_batch_size)
        trainloader = data_tmp.loader_train
        testloader = data_tmp.loader_test
    else:
        assert 1 == 0
    #
    # if dataset == 'cifar10':
    #
    #     trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
    #                                             transform=transform_train)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,drop_last=True)
    #
    #     testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False,drop_last=True)
    # elif dataset == 'cifar100':
    #     trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
    #                                              transform=transform_train)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    #
    #     testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
    #                                             transform=transform_test)
    #     testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False,drop_last=True)
    # else:
    #     assert 1 == 0

    return trainloader, testloader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def store_pruned_model(model_name,compress_rate,network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name,device=device)
    aa = copy.deepcopy(model)
    pruned_model = prune_model(aa,compress_rate,network)
    pruned_model['net'].initialize()
    save_path = Path('MyDrive/'+ model_name+'_pruned.pth')
    torch.save(pruned_model,save_path);

