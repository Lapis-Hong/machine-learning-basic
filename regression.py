# -*- coding:utf-8 -*-
from numpy import *


def load_dataset(filename):
    num_feat = len(open(filename).readline().strip('\t'))-1
    datamat, labelmat = [], []
    fr = open(filename)
    for line in fr.readlines():
        cline = map(float, line.strip().split('\t'))  # 将每一行的字符串转化为float
        datamat.append(cline[:-1])
        labelmat.append(cline[-1])
    return datamat, labelmat


def regres(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y = (y_mat - y_mat.mean()) / y_mat.std()
    X = (x_mat - x_mat.mean(0)) / x_mat.std(0)
    if linalg.det(X.T*X) == 0.0:
        print 'This matrix is singular, cannot do inverse'
    w = (X.T*X).I*X.T*y
    return w


# 局部加权 LWLR
def lwlr(test_point, x_arr, y_arr, k=1.0):
    X = mat(x_arr); y = mat(y_arr).T
    n = X.shape[0]
    W = mat(eye(n))
    for i in range(n):
        diff_vec = test_point-X[i, :]
        W[i, i] = exp(diff_vec*diff_vec.T/(-2*k**2))
    if linalg.det(X.T*W*X) == 0.0:
        print 'This matrix is singular, cannot do inverse'
    w = (X.T*W*X).I*X.T*W*y
    return test_point*w


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    y_hat = [lwlr(w, x_arr, y_arr, k) for w in test_arr]
    return map(float, y_hat)  # 将结果化为一维列表


def lwlr_plot(x_arr, y_arr, k):
    y_hat = array(lwlr_test(x_arr, x_arr, y_arr, k))  # 一维数组
    x = sort(array(x_arr), axis=0)[:, 1]
    sortind = array(x_arr)[:, 1].argsort(0)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_hat[sortind])
    ax.scatter(mat(x_arr)[:, 1].flatten().A[0], mat(y_arr).T.flatten().A[0], s=2, c='r')
    plt.show()


def rss_error(y_arr, y_hat):
    return ((y_arr-y_hat)**2).sum()


def ridge_regres(x_mat, y_mat, lam=0.2):
    denom = x_mat.T*x_mat+eye(shape(x_mat)[1])*lam
    if linalg.det(denom) == 0.0:
        print 'This matrix is singular, cannot do inverse'
    w = denom.I*x_mat.T*y_mat
    return w


def ridge_test(x, y):
    x = mat(x); y = mat(y).T
    x_scale = (x-x.mean(0))/x.std(0)
    y_scale = (y-y.mean())/y.std()
    n = 30
    w_mat = zeros((n, shape(x)[1]))
    for i in range(n):
        w = ridge_regres(x_scale, y_scale, exp(i-10))
        w_mat[i, :] = w.T
    return w_mat  # 返回array


# 逐步回归
def stage_wise(x, y, eps=0.01, num=100):
    x_mat = mat(x); y_mat = mat(y).T
    y_scale = (y_mat-y_mat.mean())/y_mat.std()
    x_scale = (x_mat-x_mat.mean(0))/x_mat.std(0)
    m = shape(x_mat)[1]
    return_mat = zeros((num, m))
    w = zeros((m, 1))
    for i in range(num):
        print w.T
        lowest_error = inf
        for j in range(m):
            for s in [-1, 1]:
                w_test = w.copy()
                w_test[j] += eps*s
                y_hat = x_scale*w_test
                rss = rss_error(y_scale.A, y_hat.A)
                if rss < lowest_error:
                    lowest_error = rss
                    w_max = w_test.copy()
        w = w_max.copy()
        return_mat[i, :] = w.T
    return return_mat


# 交叉验证岭回归
def cross_validation(x_arr, y_arr, k=10):  # k为验证次数
    n = len(y_arr)
    index = range(n)
    random.shuffle(index)
    error_mat = zeros((k, 30))
    # 划分测试集和训练集
    for i in range(k):
        x_train, x_test, y_train, y_test = [], [], [], []
        for j in range(n):
            if j < 0.9*n:
                x_train.append(x_arr[index[j]])
                y_train.append(y_arr[index[j]])
            else:
                x_test.append(x_arr[index[j]])
                y_test.append(y_arr[index[j]])
        w_mat = ridge_test(x_train, y_train)
        for l in range(30):
            x_test_mat, x_train_mat, y_train_mat = mat(x_test), mat(x_train), mat(y_train)
            x_test_scale = (x_test_mat-x_train_mat.mean(0))/x_train_mat.std(0)
            y_test_hat = (x_test_scale*mat(w_mat[l]).T)*y_train_mat.std(0)+y_train_mat.mean()
            error_mat[i, l] = rss_error(y_test_hat.T.A, array(y_test))
    mean_error = error_mat.mean(0)
    best_weight = w_mat[mean_error.argmin()]
    X, y = mat(x_arr), mat(y_arr).T
    print 'the best model from Rigde Regression is:\n', best_weight*y.std()/X.std(0)
    print 'with constant term:', y.mean()-sum(array(best_weight*y.std()/X.std(0))*array(X.mean(0)))




















