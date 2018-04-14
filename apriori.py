# -*- coding:utf-8 -*-
from numpy import *
from collections import defaultdict


def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(dataset):
    items = {item for transaction in dataset for item in transaction}
    c1 = [frozenset([w]) for w in items]  # 对每个元素构建不变集合
    c1.sort()  # 不返回
    return c1


# 剔除支持度低的项集
def scan(data, ck, min_support):
    d = map(set, data)
    item_count = defaultdict(int)
    for tran in d:
        for can in ck:
            if can.issubset(tran):
                item_count[can] += 1
    n = len(d)
    ret_list = []
    support_data = {}
    for key in item_count:
        support = item_count[key]/float(n)
        if support >= min_support:
            ret_list.append(key)
        support_data[key] = support

    return ret_list, support_data


# 由l_k-1生成c_k
def create_ck(lk):
    ret_list = []
    n = len(lk)
    for i in range(n-1):
        for j in range(i+1, n):
            l1 = list(lk[i])[:-1]
            l2 = list(lk[j])[:-1]
            l1.sort()
            l2.sort()
            if l1 == l2:
                ret_list.append(lk[i] | lk[j])
    return ret_list


# 使用apriori发现频繁项集
def apriori(dataset, min_support=0.5):
    c1 = create_c1(dataset)
    l1, support_data = scan(dataset, c1, min_support)
    l = [l1]
    k = 0
    while len(l[k]) > 0:
        ck = create_ck(l[k])
        lk, supk = scan(dataset, ck, min_support)
        support_data.update(supk)
        l.append(lk)
        k += 1
    return l, support_data


# 关联规则生成
def gene_rules(l, support_data, min_conf=0.7):
    big_rule_list = []
    for i in range(1, len(l)):
        for freq_set in l[i]:  # 每一个频繁集
            h1 = [frozenset([item]) for item in freq_set]
            if i == 1:
                calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
            else:
                h1 = calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
                rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, h, support_data, brl, min_conf=0.7):
    pruned_h = []
    for w in h:
        conf = support_data[freq_set]/support_data[freq_set-w]  # 计算可信度
        if conf > min_conf:
            print freq_set-w, '-->', w, 'conf:', conf
            brl.append((freq_set-w, w, conf))
            pruned_h.append(w)
    return pruned_h


def rules_from_conseq(freq_set, h, support_data, brl, min_conf=0.7):
    m = len(h[0])
    if len(freq_set) >= m+2:
        hmp1 = create_ck(h)  # 规则右边增加一个元素
        hmp1 = calc_conf(freq_set, hmp1, support_data, brl, min_conf)
        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)


def find_mushroom_feature(k, min_support=0.4):
    feature = str(k)
    data_set = [line.split() for line in open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch11/mushroom.dat').readlines()]
    l, support_data = apriori(data_set, min_support)
    # freq_set = [item for freq_set in l for item in freq_set if item.intersection(feature)]
    freq_dict = [item for item in support_data.items() if item[0].intersection(feature)]
    freq_dict.sort(key=operator.itemgetter(1), reverse=True)
    return freq_dict



















