# -*- coding:utf-8 -*-
from numpy import *


def load_data():
    return mat([[4, 4, 0, 2, 2],
                [4, 0, 0, 3, 3],
                [4, 0, 0, 1, 1],
                [1, 1, 1, 2, 0],
                [2, 2, 2, 0, 0],
                [1, 1, 1, 0, 0],
                [5, 5, 5, 0, 0]])


def euclid_sim(a, b):
    return 1.0/(1.0+linalg.norm(a-b))
def pearson_sim(a, b):
    if len(a) < 3:
        return 1.0
    else:
        return 0.5+0.5*corrcoef(a, b, rowvar=0)[0][1]
def cos_sim(a, b):
    num = float(a.T*b)
    denom = linalg.norm(a)*linalg.norm(b)
    return 0.5+0.5*(num/denom)


def stand_rating(data_mat, user, sim_method, item):
    n = data_mat.shape[1]
    sim_total = 0.0
    rate_total = 0.0
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0: continue
        overlap = logical_and(data_mat[:, item], data_mat[:, j]).A1
        if overlap.sum() == 0: similarity = 0
        else:
            similarity = sim_method(data_mat[overlap, item], data_mat[overlap, j])
        sim_total += similarity
        rate_total += similarity*user_rating
    if sim_total == 0: return 0
    else:
        return rate_total/sim_total


def recommend(data_mat, user, top_n=3, sim_method=cos_sim, rating_method=stand_rating):
    unrated_items = nonzero(data_mat[user, :].A1 == 0)[0]
    if unrated_items.sum() == 0:
        return 'you rated everthing'
    item_scores = []
    for item in unrated_items:
        score = rating_method(data_mat, user, sim_method, item)
        item_scores.append((item, score))
    return sorted(item_scores, key=lambda d: d[1], reverse=True)[:top_n]


def svd_rating(data_mat, user, sim_method, item):
    n = data_mat.shape[1]
    sim_total = 0.0
    rate_total = 0.0
    U, sigma, VT = linalg.svd(data_mat)
    sig4 = mat(eye(4)*sigma[:4])
    transform_items = (data_mat.T*U[:, :4]*sig4).T
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0: continue
        else:
            similarity = sim_method(transform_items[:, item], transform_items[:, j])
            sim_total += similarity
            rate_total += similarity*user_rating
    if sim_total == 0:
        return 0
    else:
        return rate_total/sim_total


def print_mat(inmat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inmat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ''


def img_compress(numsv=3, thresh=0.8):
    my_mat = mat([map(int, line.split()) for line in open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch14/0_5.txt').readlines()])
    print '*******ariginal matrix******'
    print_mat(my_mat, thresh)
    U, sigma, VT = linalg.svd(my_mat)
    recon_mat = U[:, :numsv]*mat(eye(numsv)*sigma[:numsv])*VT[:numsv, :]
    print '******reconstructed matrix using %d singular values******' % numsv
    print(recon_mat, thresh)



