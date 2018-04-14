# -*- coding:utf-8 -*-
import numpy as np
import random
import feedparser


def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classvec = [0, 1, 0, 1, 0, 1]
    return posting_list, classvec


def create_word_list(dataset):
    word_list = set()
    for w in dataset:
        word_list = word_list | set(w)
    return list(word_list)


# set-of-words model
def word2vec(word_list, inputset):
    returnvec = [0]*len(word_list)
    for word in inputset:
        if word in word_list:
            returnvec[word_list.index(word)] = 1
        else:
            print 'the word: %s is not in my Vocabulary!' % word
    return returnvec


# bag-of-words model考虑词在文档中出现的次数
def bag_word2vec(word_list, inputset):
    returnvec = [0]*len(word_list)
    for word in inputset:
        if word in word_list:
            returnvec[word_list.index(word)] += 1
        else:
            print 'the word: %s is not in my Vocabulary!' % word
    return returnvec


def train_nb(train_mat, train_class):
    numdocs = len(train_mat)
    numwords = len(train_mat[0])
    pabusive = sum(train_class)/float(numdocs)
    #拉普拉斯平滑，防止概率相乘中的某一项为0导致乘积为0,构建向量
    p0num = np.ones(numwords); p1num = np.ones(numwords)
    p0denom =2.0; p1denom = 2.0
    for i in range(numdocs):
        if train_class[i] == 1:
            p1num += train_mat[i]
            p1denom += sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0denom += sum(train_mat[i])
    # 求p(wi|c0)以及p(wi|c1),log为了解决下溢出，太多小的数相乘易四舍五入为0，
    p1 = np.log(p1num/p1denom); p0 = np.log(p0num/p0denom)
    return p0, p1, pabusive


def classify_nb(classify_vec, p0, p1, pclass1):
    # 连乘积p(w|c1) *p(c1)取对数
    p0 = sum(classify_vec*p0) + np.log(1-pclass1)
    p1 = sum(classify_vec*p1) + np.log(pclass1)
    return np.where(p0 >p1, 0, 1)

def testing_nb():
    posts, classes = load_dataset()
    vocab_list = create_word_list(posts)
    train_mat = []
    for w in posts:
        train_mat.append(word2vec(vocab_list, w))
    p0, p1, pabusive = train_nb(np.array(train_mat), np.array(classes))
    test_entry = ['love', 'my', 'I']
    this_doc = word2vec(vocab_list, test_entry)
    print test_entry, 'classified as:', classify_nb(this_doc, p0, p1, pabusive)


# 文件解析以及完整的垃圾邮件测试函数
def text_parse(string):
    import re
    # 分词，过滤
    tokens = re.split(r'\W*', string)
    return [tok.lower() for tok in tokens if len(tok) > 2]


def spam_test(n):
    error = []
    for i in range(n):
        doc_list = []; class_list = []; full_text = []
        for j in range(1, 26):
            word_list = text_parse(open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch04/email/spam/%d.txt' % j).read())
            doc_list.append(word_list)  # 列表中列表
            full_text.extend(word_list)  # z只有一个列表
            class_list.append(1)
            word_list = text_parse(open('/home/hongquan/Documents/mlia code/machinelearninginaction/Ch04/email/ham/%d.txt' % j).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        vocab_list = create_word_list(doc_list)
        # 随机抽样，构建训练测试集
        train_set = random.sample(range(50), 40); test_set = np.setdiff1d(range(50), train_set)
        train_mat = []; train_class = []
        for w in train_set:
            train_mat.append(bag_word2vec(vocab_list, doc_list[w]))
            train_class.append(class_list[w])
        p0, p1, pspam = train_nb(np.array(train_mat), np.array(train_class))
        error_count = 0.0
        for w in test_set:
            word_vec = bag_word2vec(vocab_list, doc_list[w])
            if classify_nb(np.array(word_vec), p0, p1, pspam) != class_list[w]:
                error_count += 1
                print 'classification error', doc_list[w]
        print 'the error rate is:', error_count/len(test_set)
        error.append(error_count/len(test_set))
    print np.mean(error)


def most_freq(vocab_list, full_text):
    freqdict = {}
    for token in vocab_list:
        freqdict[token] = full_text.count(token)
        sorted_freq = sorted(freqdict.iteritems(), key=lambda d: d[1], reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list = []; class_list = []; full_text = []
    minlen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minlen):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_word_list(doc_list)
    # 高频词去除
    top30words = most_freq(vocab_list, full_text)
    for w in top30words:
        if w[0] in vocab_list:
            vocab_list.remove(w[0])
    train_set = random.sample(range(2*minlen), 20); test_set = np.setdiff1d(range(2*minlen), train_set)
    train_mat = []; train_class = []
    for w in train_set:
        train_mat.append(bag_word2vec(vocab_list, doc_list[w]))
        train_class.append(class_list[w])
    p0, p1, pspam = train_nb(np.array(train_mat), np.array(train_class))
    error_count = 0.0
    for w in test_set:
        word_vec = bag_word2vec(vocab_list, doc_list[w])
        if classify_nb(np.array(word_vec), p0, p1, pspam) != class_list[w]:
            error_count += 1
    print 'the error rate is:', error_count / len(test_set)
    return vocab_list, p0, p1


# 显示最具有表征性的词汇,p(wi|ci)条件概率最大的
def gettopword(ny, sf):
    vocab_list, p0, p1 = local_words(ny, sf)
    NY =[]; SF = []
    for i in range(len(p0)):
        NY.append((vocab_list[i], p1[i]))
        SF.append((vocab_list[i], p0[i]))
    sortedNY = sorted(NY, key=lambda d: d[1], reverse=True)
    sortedSF = sorted(SF, key=lambda d: d[1], reverse=True)
    print 'NY type words:'
    for n in sortedNY[:10]:
        print n[0],
    print '\n', 'SF type words:'
    for n in sortedSF[:10]:
        print n[0],


ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')






















