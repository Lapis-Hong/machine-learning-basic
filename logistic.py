# -*- coding:utf-8 -*-
from numpy import *
def load_dataset():
    data_mat, label_mat = [], []
    fr = open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(x):
    return 1.0/(1+exp(longfloat(-x)))  # 避免指数溢出


# 梯度上升法
def grad_ascent(data, labels, alpha=0.001, k=500):
    data_mat = mat(data)
    label_mat = mat(labels).T
    n = data_mat.shape[1]
    w = ones((n, 1))
    for i in range(k):
        h = sigmoid(data_mat*w)
        error = label_mat-h
        w += alpha*data_mat.T*error
    return w


def plot_fit(w):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_dataset()
    # w = grad_ascent(data_mat, label_mat)
    n = shape(data_mat)[0]
    data_arr = array(data_mat)
    class1, class2 = [], []
    for i in range(n):
        if label_mat[i] == 1:
            class1.append(data_arr[i, 1:3])
        else:
            class2.append(data_arr[i, 1:3])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array(class1)[:, 0], array(class1)[:, 1], c='red', marker='s')
    ax.scatter(array(class2)[:, 0], array(class2)[:, 1], c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1', fontsize='small')
    plt.ylabel('X2', fontsize='large')
    plt.show()


# 随机梯度上升法
def stoc_grad_ascent(data, labels, alpha=0.01, k=150):
    data_arr = array(data)
    m, n = shape(data)
    w = ones(n)
    for i in range(k):
        for j in range(m):
            data_index = range(m)
            alpha = 4/(1.0+i+j)+0.01
            rand_index = int(random.uniform(0, len(data)))
            h = sigmoid(sum(data_arr[rand_index] * w))
            error = labels[rand_index] - h
            w += alpha * data_arr[rand_index] * error
            del data_index[rand_index]
    return w


# logistic分类回归
def classify(x, weights):
    prob = sigmoid(sum(x*weights))
    return 1.0 if prob > 0.5 else 0.5


def colic_test():
    fr_train = open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch05/horseColicTraining.txt')
    fr_test = open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch05/horseColicTest.txt')
    train_set, train_label = [], []
    for line in fr_train.readlines():
        line = map(float, line.strip().split('\t'))
        train_set.append(line)
        train_label.append(line[21])
    train_weights = stoc_grad_ascent(array(train_set), train_label, k=500)
    error_count = 0
    total_count = 0
    for line in fr_test.readlines():
        total_count += 1
        line = map(float, line.strip().split('\t'))
        if int(classify(array(line), train_weights)) != int(line[21]):
            error_count += 1
    error_rate = float(error_count)/total_count
    print 'the error rate is: %f' % error_rate
    return error_rate


# 测试k次求平均错误率
def multitest(k):
    error_sum = 0.0
    for k in range(k):
        error_sum += colic_test()
    print 'after %d iterations the average error rate is: %f' % (k, error_sum/k)








