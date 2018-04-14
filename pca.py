# -*- coding:utf-8 -*-
from numpy import *


path = '/home/hongquan/Documents/mlia code/machinelearninginaction/Ch13'
def load_dataset(file_name, delim='\t'):
    fr = open(file_name)
    dat_arr = [map(float, line.strip().split(delim)) for line in fr.readlines()]
    return mat(dat_arr)


def percentage2n(eigvals, percentage):
    arr = sort(eigvals)
    arr_reverse = arr[::-1]
    eig_cumsum = 0
    n = 0
    for eig in arr_reverse:
        eig_cumsum += eig
        n += 1
        if eig_cumsum > percentage*arr_reverse.sum():
            return n


def pca(data_mat, percentage=0.99):
    mean_removed = data_mat-data_mat.mean(0)
    cov_mat = cov(mean_removed, rowvar=0)
    eigvals, eigvecs = linalg.eig(mat(cov_mat))
    n = percentage2n(eigvals, percentage)
    eigval_ind = eigvals.argsort()
    eigval_ind = eigval_ind[:-(n+1):-1]
    red_vec = eigvecs[:, eigval_ind]
    red_mat = mean_removed*red_vec
    recon_mat = (red_mat*red_vec.T)+data_mat.mean(0)  # 重构数据
    return sort(eigvals), red_mat, recon_mat


def replace_with_mean():
    dat_arr = array(load_dataset('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch13/secom.data', ' '))
    num_feat = dat_arr.shape[1]
    for i in range(num_feat):
        mean_val = dat_arr[~isnan(dat_arr[:, i]), i].mean()
        dat_arr[isnan(dat_arr[:, i]), i] = mean_val
    return mat(dat_arr)



