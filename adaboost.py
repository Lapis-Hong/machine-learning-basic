# -*- coding:utf-8 -*-
from numpy import *


def load_simple_data():
    datmat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    class_label = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datmat, class_label


#  单层决策树生成q
def stump_classify(datamat, feature, thresh_val, thresh_ineq):
    class_label = ones((len(datamat), 1))
    if thresh_ineq == 'lt':
        class_label[datamat[:, feature] <= thresh_val] = -1.0
    else:
        class_label[datamat[:, feature] > thresh_val] = -1.0
    return class_label


def build_stump(data_arr, labels, weights, num_steps=10):
    data_mat = mat(data_arr)
    label_mat = mat(labels).T
    m, n = shape(data_mat)
    best_stump = {}
    min_error = inf
    for i in range(n):  # 遍历特征
        min_val = data_mat[:, i].min()
        step_size = (data_mat[:, i].max()-data_mat[:, i].min())/num_steps
        for j in range(num_steps+1):
            for inequal in ['lt', 'gt']:
                thresh_val = min_val + float(j)*step_size
                predicted_label = stump_classify(data_mat, i, thresh_val, inequal)
                err_vec = mat(ones((m, 1)))
                err_vec[predicted_label == label_mat] = 0
                weighted_error = weights.T*err_vec
                # print 'split: dim %d, thresh %.2f, thresh_ineq: %s, the weighted error is %.3f: ' % (i, thresh_val, inequal, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_label = predicted_label.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, best_label, float(min_error)


def adaboost_train(data_arr, labels, k=30):
    class_arr = []
    m = data_arr.shape[0]
    w = mat(ones((m, 1))/m)
    agg_class = mat(zeros((m, 1)))
    for i in range(k):
        best_stump, weak_class, error = build_stump(data_arr, labels, w)
        # print 'w:', w.T
        alpha = 0.5*log((1-error)/(max(error, 1e-16)))
        best_stump['alpha'] = alpha
        class_arr.append(best_stump)
        # print 'class:', weak_class.T
        w = multiply(w, exp(multiply(-alpha*mat(labels).T, weak_class)))
        w = w/w.sum()
        agg_class += alpha*weak_class
        # print 'agg_class:', agg_class.T
        agg_error = multiply(sign(agg_class) != mat(labels).T, ones((m, 1)))
        error_rate = agg_error.sum()/m
        # print 'total error:', error_rate, '\n'
        if error_rate == 0.0: break
    return class_arr, agg_class


def ada_classify(test_mat, class_arr):
    data_mat = mat(test_mat)
    m = data_mat.shape[0]
    agg_class = mat(zeros((m, 1)))
    for i in range(len(class_arr)):
        weak_class = stump_classify(data_mat, class_arr[i]['dim'], class_arr[i]['thresh'], class_arr[i]['ineq'])
        agg_class += class_arr[i]['alpha']*weak_class
        print agg_class
    return sign(agg_class)


def load_dataset(file_name):
    data_mat = []
    for line in open(file_name).readlines():
        line = map(float, line.strip().split('\t'))
        data_mat.append(line)
    return data_mat


def plot_roc(pred_strengths, labels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ysum = 0.0
    num_positive = sum(array(labels)==1.0)
    ystep = 1/float(num_positive)
    xstep = 1/float(len(labels)-num_positive)
    sorted_index = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    for index in sorted_index.tolist()[0]:
        if labels[index] == 1.0:
            delx = 0
            dely = ystep
        else:
            delx = xstep
            dely = 0
            ysum += cur[1]
        ax.plot([cur[0], cur[0]-delx], [cur[1], cur[1]-dely], c='b')
        cur = (cur[0]-delx, cur[1]-dely)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        ax.axis([0, 1, 0, 1])
        plt.show()
    print 'the area under the curve is:', ysum*xstep













