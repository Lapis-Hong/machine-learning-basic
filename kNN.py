from __future__ import division
import numpy as np
import pandas as pd
import operator
from os import listdir
import matplotlib.pyplot as plt

def createdataset():
    groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return groups, labels


def classify(x, dataset, labels, k):
    diffmat = dataset-x
    dist = np.sqrt((diffmat**2).sum(axis=1))
    distindex = dist.argsort()
    # k labels count
    class_count = {}
    for i in range(k):
        ilabel = labels[distindex.tolist()[i]]
        class_count[ilabel] = class_count.get(ilabel, 0)+1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]


df = pd.read_table('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch02/datingTestSet.txt', header=None)
datinglabels = df.iloc[:, 3]
labels_num = datinglabels.replace({'didntLike':1, 'smallDoses':2, 'largeDoses': 3}).tolist()
#labels_num = datinglabels.where(datinglabels=='didntLike', 1.0, datinglabels.where(datinglabels=='smallDoses', 2.0, 3.0))
datingdata = df.iloc[:, :3]
normdata = (datingdata-datingdata.min(0))/(datingdata.max(0)-datingdata.min(0))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x=datingdata.iloc[:, 1], y=datingdata.iloc[:, 2],  c=15.0*np.array(labels_num))
plt.show()


def datingtest():
    ratio = 0.1
    m = len(normdata) #m=normdata.shape[0]
    numtest = int(ratio*m)
    error_count = 0.0
    for i in range(numtest):
        classfier_result = classify(normdata.ix[i].tolist(), normdata.ix[numtest:m], labels_num[numtest:m], 50)
        print 'the classifier is: %d, the real is : %d' % (classfier_result, labels_num[i])
        if classfier_result != labels_num[i]:
            error_count += 1.0
    print 'the total error rate is: %f' % (error_count/float(numtest))


def img2vec(filename):
    returnvec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvec[0,32*i+j] = int(linestr[j])
    return returnvec


def handwriting_test():
    hwlabels = []
    training_file_list = listdir('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch02/trainingDigits')
    m = len(training_file_list)
    trainingmat = np.zeros((m, 1024))
    for i in range(m):
        file_name = training_file_list[i]
        realnum = int(file_name[0])
        hwlabels.append(realnum)
        trainingmat[i, :] = img2vec('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch02/trainingDigits/%s' % file_name)

    test_file_list = listdir('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch02/testDigits')
    mtest = len(test_file_list)
    error_count = 0.0
    for i in range(mtest):
        file_name = test_file_list[i]
        realnum = int(file_name[0])
        test_vec = img2vec('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch02/testDigits/%s' % file_name)
        classify_result = classify(test_vec, trainingmat, hwlabels, 3)
        print 'the classifier is : %d, the real answer is %d' % (classify_result, realnum)
        if classify_result != realnum:
            error_count += 1.0
    print '\nthe total number of errors is: %d' % error_count
    print '\nthe total number of errors rate is: %f' % (error_count/mtest)
















