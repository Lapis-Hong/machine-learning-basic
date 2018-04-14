# -*- coding:utf-8 -*-
from math import log
import operator
import numpy as np
from matplotlib import pyplot as plt


def shannon_ent(dataset):
    num = len(dataset)
    label_counts = {}
    # 为所有可能分类创建字典，值为次数
    for vec in dataset:
        current_label = vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] = 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# 按照给定特征划分数据集
def split_dataset(dataset, axis, value):
    retdataset = []
    for vec in dataset:
        if vec[axis] == value:
            reduced_vec = vec[:axis] + vec[axis+1:]
            retdataset.append(reduced_vec)
    return retdataset


# 选择最好的特征
def best_feature(dataset):
    num_features = len(dataset[0])-1
    base_ent = shannon_ent(dataset)
    best_info_gain = 0.0; best_feat = -1
    # 遍历特征
    for i in range(num_features):
        # 取多维列表第i个元素
        feat_list = [w[i] for w in dataset]
        unique_vals = set(feat_list)
        # 计算特征i的条件熵
        new_ent = 0.0
        for val in unique_vals:
            sub_dataset = split_dataset(dataset, i, val)
            prob = len(sub_dataset)/float(len(dataset))
            new_ent += prob*shannon_ent(sub_dataset)
        info_gain = base_ent-new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat


# 多数表决决定叶子节点分类
def majority(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 创建树
def create_tree(dataset, labels):
    class_list = [w[-1] for w in dataset]  # 所有类标签
    if len(set(class_list)) == 1:  # 类别相同停止划分
        return class_list[0]
    if len(dataset[0]) == 1:  # 使用完所有特征
        return majority(class_list)
    # 递归树
    best_feat = best_feature(dataset)
    best_feature_label = labels[best_feat]
    my_tree = {best_feature_label: {}}
    del labels[best_feat]
    feat_val = [w[best_feat] for w in dataset]
    unique_val = set(feat_val)
    for val in unique_val:
        sublabels = labels[:]
        #sublabels = labels[:best_feat] + labels[best_feat + 1:]  # 不改变输入参数labels
        my_tree[best_feature_label][val] = create_tree(split_dataset(dataset, best_feat, val), sublabels)
    return my_tree


decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(nodetxt, centerpt, parentpt, nodetype):
    create_plot.ax1.annotate(nodetxt, xy=parentpt, xycoords='axes fraction', xytext=centerpt, textcoords='axes fraction',
                             va='center', ha='center', bbox=nodetype, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('internal node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leaf(my_tree):
    num_leafs = 0
    second_dict = my_tree.values()[0]
    for key in second_dict.keys():
        if type(second_dict[key]) == dict:
            num_leafs += get_num_leaf(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    max_depth = 0
    second_dict = my_tree.values()[0]
    for key in second_dict.keys():
        if type(second_dict[key]) == dict:
            this_depth = 1+get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

# my_tree = {'flippers': {0: 'no', 1: {'no surfacing': {0: 'no', 1: 'yes'}}}}

# 使用决策树分类
def classify(tree, labels, testvec):
    first_str = tree.keys()[0]
    second_dict = tree.values()[0]
    feat_index = labels.index(first_str)
    for key in second_dict.keys():
        if testvec[feat_index] == key:
            if type(second_dict[key]) == dict:
                class_label = classify(second_dict[key], labels, testvec)
            else:
                class_label = second_dict[key] #此时叶结点
    return class_label


# 使用pickle模块存储决策树
def store_tree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()
def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# 隐形眼睛数据
fr = open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch03/lenses.txt')
lenses = [s.strip().split('\t') for s in fr.readlines()]
lenses_label = ['age', 'prescript', 'astigmatic', 'tearrate']












