# -*- coding:utf-8 -*-
from numpy import*


def load_dataset(file_name):
    data_mat = []
    fr = open(file_name)
    for line in  fr.readlines():
        line = map(float, line.strip().split('\t'))
        data_mat.append(line)
    return mat(data_mat)


def dist_eclud(vec1, vec2):
    return sqrt(sum(power(vec1-vec2, 2)))


#  构建k个随机质心
def rand_center(dataset, k):
    m = shape(dataset)[1]
    center = mat(zeros((k, m)))
    for j in range(m):
        min_j = dataset[:, j].min()
        range_j = float(dataset[:, j].max()-min_j)
        center[:, j] = min_j + range_j*random.rand(k, 1)
    return center


def kMeans(dataset, k, dist_method=dist_eclud, create_center=rand_center):
    n = shape(dataset)[0]
    center = create_center(dataset, k)
    cluster_result = mat(zeros((n, 2)))  # 聚类结果第一列储存类别， 第二列储存距离
    cluster_changed = True
    while cluster_changed:  # 当类别没有改变时停止循环
        cluster_changed = False
        for i in range(n):
            min_dist = inf
            min_index = -1
            for j in range(k):  # 对第i个样本，找出最近的质心
                dist_j = dist_method(center[j, :], dataset[i, :])
                if dist_j < min_dist:
                    min_dist = dist_j
                    min_index = j
            if cluster_result[i, 0] != min_index: cluster_changed = True
            cluster_result[i, :] = min_index, min_dist**2
        # print center
        for j in range(k):
            cluster_j = dataset[nonzero(cluster_result[:, 0] == j)[0]]
            center[j, :] = mean(cluster_j, axis=0)  # 更新质心的位置
    return center, cluster_result


# 二分 kmeans
def bikMeans(dataset, k, dist_method=dist_eclud):
    n = shape(dataset)[0]
    clu_result = mat(zeros((n, 2)))  # 每个样本簇的类别加上距离该质心的距离
    center0 = dataset.mean(0).tolist()[0]  # 初始簇质心
    center_list = [center0]  # 创建质心列表
    for j in range(n):
        clu_result[j, 1] = dist_method(mat(center0), dataset[j, :])**2
    while len(center_list) < k:  # 当簇个数到达k后停止
        min_sse = inf
        for i in range(len(center_list)):  # 对每个簇遍历，用kmeans二分，计算SSE
            ind_i = nonzero(clu_result[:, 0] == i)[0]
            ind_noni = nonzero(clu_result[:, 0] != i)[0]
            center, bi_clu_result = kMeans(dataset[ind_i], 2, dist_method)
            sse_split = bi_clu_result[:, 1].sum()
            sse_nonsplit = clu_result[ind_noni, 1].sum()
            print 'seespli, and notsplit:', sse_split, sse_nonsplit
            if (sse_split+sse_nonsplit) < min_sse:
                min_sse = sse_split + sse_nonsplit  # 划分哪一个使其最大程度降低总的SSE
                which_clu = i  # 选择第i个簇二分
                bi_center = center  # 有两个新的质心
                bi_clu_result = bi_clu_result.copy()  # 二分结果
        bi_clu_result[nonzero(bi_clu_result[:, 0] == 0)[0], 0] = which_clu  # 更新簇的分配结果
        bi_clu_result[nonzero(bi_clu_result[:, 0] == 1)[0], 0] = len(center_list)
        center_list[which_clu] = bi_center[0]  # 用二分后欺诈其中一个质心代替原来的第i个
        center_list.append(bi_center[1])  # 增加一个质心
        clu_result[nonzero(clu_result[:, 0] == which_clu)[0]] = bi_clu_result
    return center_list, clu_result

import urllib, json
def geo_grab(staddress, city):
    api = 'http://where.yahooapis.com/geocode?'
    params = {'flags':  'J', 'appid': 'ppp68N8t', 'location': '%s %s' % (staddress, city)}
    url_params = urllib.urlencode(params)
    yahoo_api = api + url_params
    print yahoo_api
    return json.loads(urllib.urlopen(yahoo_api).read())


from time import sleep
def place_find(file_name):
    fw = open('places.txt', 'w')
    for line in open(file_name).readlines():
        line = line.strip()
        line_arr = line.split('\t')
        ret_dict = geo_grab(line_arr[1], line_arr[2])
        if ret_dict['ResultSet']['Error'] == 0:
            lat = float(ret_dict['ResultSet']['Results'][0]['latitude'])
            lon = float(ret_dict['ResultSet']['Results'][0]['longitude'])
            print '%s\t%f\t%f' %(line_arr[0], lat, lon)
            fw.write('%s\t%f\t%f\n' % (line, lat, lon))
        else: print 'error fetching'
        sleep(1)
    fw.close()











