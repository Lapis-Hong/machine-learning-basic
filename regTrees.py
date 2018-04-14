# -*- coding:utf-8 -*-
from numpy import *


def load_dataset(file_name):
    # data_mat = []
    # fr = open(file_name)
    # for line in fr.readlines():
    #     line = map(float, line.strip().split('\t'))
    #     data_mat.append(line)
    # return data_mat
    data_list = [map(float, line.strip().split('\t')) for line in open(file_name).readlines()]
    return array(data_list)


def bi_split_dataset(dataset, feature, value):
    arr0 = dataset[dataset[:, feature] > value]
    arr1 = dataset[dataset[:, feature] <= value]
    return arr0, arr1


def reg_leaf(dataset):
    return mean(dataset[:, -1])


def reg_error(dataset):
    return len(dataset)*var(dataset)


def choose_best_split(dataset, leaf_type=reg_leaf, err_type=reg_error, ops=(1, 4)):
    tol_e = ops[0]
    tol_n = ops[1]
    if len(set(dataset[:, -1])) == 1:
        return None, leaf_type(dataset)
    e = err_type(dataset)
    best_e, best_feat, best_val = inf, 0, 0
    for feat in range(dataset.shape[1]-1):  # 遍历特征
        for val in set(dataset[:, feat]):  # 遍历特征feat的所有取值
            arr0, arr1 = bi_split_dataset(dataset, feat, val)
            if len(arr0) < tol_n or len(arr1) < tol_n:
                continue
            new_e = err_type(arr0) + err_type(arr1)
            if new_e < best_e:
                best_e = new_e
                best_feat = feat
                best_val = val
    if e - best_e < tol_e:
        return None, leaf_type(dataset)
    arr0, arr1 = bi_split_dataset(dataset, best_feat, best_val)
    if len(arr0) < tol_n or len(arr1) < tol_n:
        return None, leaf_type(dataset)

    return best_feat, best_val


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_error, ops=(1, 4)):
    split_feat, split_val = choose_best_split(dataset, leaf_type, err_type, ops)
    if split_feat is None: return split_val
    ret_tree = {'split_ind': split_feat, 'split_val': split_val}  # 用字典储存树结构
    lset, rset = bi_split_dataset(dataset, split_feat, split_val)
    ret_tree['left'] = create_tree(lset, leaf_type, err_type, ops)  # 递归调用
    ret_tree['right'] = create_tree(rset, leaf_type, err_type, ops)
    return ret_tree


def is_tree(obj):
    return type(obj) == dict


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return 0.5*(tree['right']+tree['left'])


def prune(tree, test_data):
    if len(test_data) == 0:
        return get_mean(tree)  # 如果测试数据为空，返回树平均值
    if is_tree(tree['right']) or is_tree(tree['left']):  # 存在至少一个子树
        lset, rset = bi_split_dataset(test_data, tree['split_ind'], tree['split_val'])
        if is_tree(tree['left']): tree['left'] = prune(tree['left'], lset)  # 递归左树剪枝
        if is_tree(tree['right']): tree['right'] = prune(tree['right'], rset)
    if not is_tree(tree['left']) and not is_tree(tree['right']):  # 两个分支已经都是叶节点
        lset, rset = bi_split_dataset(test_data, tree['split_ind'], tree['split_val'])
        error_nomerge = sum((lset[:, -1]-tree['left'])**2) + sum((rset[:, -1]-tree['right'])**2)
        tree_mean = 0.5*(tree['right']+tree['left'])
        error_merge = sum((test_data[:, -1]-tree_mean)**2)
        if error_merge < error_nomerge:
            print 'merging'
            return tree_mean
        else:
            return tree
    else:
        return tree


def linear_solve(dataset):
    x = mat(dataset[:, :-1])
    y = mat(dataset[:, -1]).T
    xTx = x.T*x
    if linalg.det(xTx) == 0.0:
        raise NameError('this matrix is singular, can not do inverse')
    w = xTx.I*x.T*y
    return w, x, y


def model_leaf(dataset):
    w, x, y = linear_solve(dataset)
    return w


def model_error(dataset):
    w, x, y = linear_solve(dataset)
    return (y-x*w).T*(y-x*w)







