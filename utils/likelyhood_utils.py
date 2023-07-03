import operator
import glob
from torch import nn
from collections import Counter,OrderedDict
import torch
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt

def get_dist_matrix(layer_grad, componets):
    filter_num = layer_grad.shape[0]
    grad_tensor = torch.sum(layer_grad, dim=1).view(filter_num, -1)  # 可以试试mean等效果如何
    pca = PCA(n_components=componets)
    grad_tensor_reduced = pca.fit_transform(grad_tensor.cpu().numpy())  # x还是最开始那个x

    grad_tensor_reduced = torch.from_numpy(grad_tensor_reduced).cuda()

    grad_dist_matrix = torch.norm(grad_tensor_reduced[:, None] - grad_tensor_reduced, dim=2, p=2)
    return grad_dist_matrix

def get_model(model_name, device='cuda:0'):
    if sys.platform == 'linux':
        pth_path = './MyDrive/*.pth'
    else:
        # project_root_path = os.path.abspath(os.path.dirname(__file__))
        ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        pth_path = ROOT_DIR + os.sep + '*' + os.sep + '*.pth'
    # for file in glob.glob(pth_path):
    #     if model_name in file:
    #         model = torch.load(file, map_location='cuda:0')
    #         model = nn.DataParallel(model, device_ids=device)
    #         # model = model.cuda(device=device[0])
    #         return model.module
    for file in glob.glob(pth_path):
        if model_name in file:
            model = torch.load(file,map_location='cuda:0')
            return model


#传入距离矩阵分分位数
def get_likely_point(grad_dist_matrix,point):
    #np.percentile去相异度矩阵的中位数
    #grad_dist_matrix<中位数代表相异度小的卷积核也就是相似度大的卷积核序号
    likely_ndarray = np.where(grad_dist_matrix.cpu().numpy() < np.percentile(grad_dist_matrix.cpu().numpy(), point))
    return likely_ndarray

def get_corinates_count(likely_ndarray,dic_a):
    list0 = likely_ndarray[0]
    list1 = likely_ndarray[1]
    list0 = list(map(str, list0))
    list1 = list(map(str, list1))
    # list0 = ['0', '2', '3', '4', '5', '6', '8', '7']
    # list1 = ['1', '3', '4', '3', '4', '5', '6', '1']
    for idx, i in enumerate(list0):
        str_a = i + '+' + list1[idx]
        str_reverse = list1[idx] + '+' + i
        if str_a == str_reverse:
            continue
        if str_a not in dic_a.keys() and str_reverse not in dic_a.keys():
            dic_a[str_a] = 1
        elif str_a in dic_a.keys():
            dic_a[str_a] = dic_a[str_a] + 1
        elif str_reverse in dic_a.keys():
            dic_a[str_reverse] = dic_a[str_reverse] + 1
    return dic_a


#接收相似度大于某分位数的筛选结果，返回卷积核出现次数统计的次数，likely_ndarray有两个数组，内容其实是一样的，统计其中一个就行，
def get_filter_index(likely_ndarray):
    return Counter(likely_ndarray[0])


def get_count_statics(count,num):
    list_idx_freq = count['layer10_conv'].most_common(num)
    return list(map(lambda x:x[0],list_idx_freq)),list(map(lambda x:x[1],list_idx_freq))



def get_count_dict(net, compress_rate, count_dict):

    for layer_name, layer in net._modules.items():
        #resnet的下采样层和卷积核尺寸小于4的不处理
        if isinstance(layer, nn.Conv2d) and 'downsample' not in layer_name:
            componet = 4 if layer.kernel_size[0] > 2 else 1
            grad_dist_matrix = get_dist_matrix(layer.weight.grad, componet)
            # visualize_dist_matrix(grad_dist_matrix)#矩阵可视化
            likely_ndarray = get_likely_point(grad_dist_matrix, (1 - compress_rate) * 100)
            filter_cpunt = get_filter_index(likely_ndarray)
            try:
                count_dict[layer_name] += filter_cpunt
            except KeyError:
                count_dict[layer_name] = filter_cpunt
        # grad_dist_matrix = get_dist_matrix(net, 4)
        #
        # likely_ndarray = get_likely_point(grad_dist_matrix, (1 - args.compress_rate) * 100)
        # filter_count += get_filter_index(likely_ndarray)
    return count_dict







def visualize_dist_matrix(input):
    input = input.cpu().numpy()
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    fig, ax = plt.subplots()
    im = ax.imshow(input)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    #Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ##Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, input[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()
