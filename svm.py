# -*- coding:utf-8 -*-
from numpy import *


def load_dataset(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def j_rand(i, m):
    j = i
    while j == i:
        j = random.randint(0, m)
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smo_simple(data, labels, C, tol, max_iter):
    x = mat(data)
    y = mat(labels).T
    b = 0
    m = x.shape[0]
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            x_i = x[i, :]
            fx_i = float(multiply(alphas, y).T*(x*x_i.T))+b
            E_i = fx_i-float(y[i])
            if (y[i]*E_i < -tol and alphas[i] < C) or (y[i]*E_i > tol and alphas[i] > 0):
                j = j_rand(i, m)
                x_j = x[j, :]
                fx_j = float(multiply(alphas, y).T*(x*x_j.T))+b
                E_j = fx_j - float(y[j])
                alpha_iold = alphas[i].copy()
                alpha_jold = alphas[j].copy()
                if y[i] != y[j]:
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    print 'L==H'
                    continue
                eta = -(x_i-x_j)*(x_i-x_j).T
                if eta >= 0:
                    print 'eta>=0'
                    continue
                alphas[j] -= y[j]*(E_i-E_j)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j]-alpha_jold < 0.00001):
                    print 'j not moving enough'
                    continue
                alphas[i] += y[j]*y[i]*(alpha_jold-alphas[j])
                b1 = b-E_i-y[i]*(alphas[i]-alpha_iold)*x_i*x_i.T-y[j]*(alphas[j]-alpha_jold)*x_i*x_j.T
                b2 = b-E_j-y[i]*(alphas[i]-alpha_iold)*x_i*x_i.T-y[j]*(alphas[j]-alpha_jold)*x_i*x_j.T
                if (0 < alphas[i]) and (alphas[i] < C):
                    b = b1
                elif (0 < alphas[j]) and (alphas[j] < C):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alpha_pairs_changed += 1
                print 'iter:%d i:%d pairs changed: %d' %(iter, i, alpha_pairs_changed)
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            print 'iteration number: %d' % iter
    return b, alphas









