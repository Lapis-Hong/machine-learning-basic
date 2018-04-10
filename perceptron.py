#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/4/10
import numpy as np
import matplotlib.pyplot as plt


class ShowPicture(object):
    def __init__(self, data, w, b):
        self.b = b
        self.w = w
        plt.figure(1)
        plt.title('Plot 1', size=14)
        plt.xlabel('x-axis', size=14)
        plt.ylabel('y-axis', size=14)

        x = np.linspace(0, 5, 100)
        y = self.expression(x)
        plt.plot(x, y, color='r', label='y1 data')

        plt.scatter(data[0][0], data[0][1], s=50)
        plt.scatter(data[1][0], data[1][1], s=50)
        plt.scatter(data[2][0], data[2][1], marker='x', s=50, )
        plt.savefig('2d.png', dpi=75)

    def expression(self, x):
        y = (-self.b - self.w[0] * x) / self.w[1]
        return y

    def show(self):
        plt.show()


class Perceptron(object):

    def __init__(self):
        self._w = None
        self._b = None

    def fit(self, x, y, lr=0.01, step=1000):
        """Each step use the worst misclassfication point to update weights."""
        # when data is already array, np.asarray does not copy, but np.array does.
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0
        for _ in range(step):
            # Loss function L(w, b) = -sum y_i*(w*x_i+b), misclassification point: -y(w*x+b)>0
            y_pred = x.dot(self._w) + self._b  # Xw+b
            # chose max loss point, if this is right, all have been classify right
            idx = np.argmax(np.maximum(0, -y_pred * y))
            if y[idx] * y_pred[idx] > 0:
                break
            # update weight
            self._w += lr * y[idx] * x[idx]
            self._b += lr * y[idx]

    def fit2(self, x, y, lr=0.1):
        """upaate weights in order."""
        flag = True
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0

        while flag:
            count = 0
            print(self._w)
            for i in range(len(y)):
                y_pred = np.dot(x[i], self._w) + self._b
                print(y_pred)
                np.dot(x[i, :], self._w)
                if y_pred * y[i] <= 0:
                    self._w += lr * y[i] * x[i]  # x[i] == x[i, :]
                    self._b += y[i]
                    count += 1
            if count == 0:  # no misclassification points
                flag = False

    def get_weight(self):
        return self._w, self._b

    def predict(self, x):

        return np.where(np.dot(np.asarray(x), self._w) + self._b >= 0, 1, -1)


if __name__ == '__main__':
    x = [(3, 3), (4, 3), (1, 1)]
    y = [1, 1, -1]
    model = Perceptron()
    model.fit(x, y)
    w, b = model.get_weight()
    y = model.predict([3, 4])
    print('prediction label: {}'.format(y))
    picture = ShowPicture(x, w, b)
    picture.show()


