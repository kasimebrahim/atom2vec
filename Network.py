"""
9/18/17
kasim
se.kasim.ebrahim@gmail.com
"""

import numpy as np
from functions import softmax, cross_entropy


class Network:
    def __init__(self, input_size, feature_size):
        self.input_size = input_size
        self.feature_size = feature_size

        self.weights = np.random.randn(self.feature_size, self.input_size)
        self.features = np.random.randn(self.input_size, self.feature_size)

        # print self.weights.shape, self.input_size

    def forward(self, _input):
        z = np.dot(self.weights, _input)
        g = np.dot(self.features, z)
        s = softmax.func(g)
        # print g
        # print np.sum(s)
        return z, g, s

    def feed_forward(self, x_mini_batch, y_mini_batch):
        jacobian = np.zeros((self.input_size, self.feature_size))
        totat_cost = 0;
        for x, y in zip(x_mini_batch, y_mini_batch):
            _x = x.reshape(x.size, 1)
            _y = y.reshape(y.size, 1)
            # print _x, _y
            z, g, s = self.forward(_x)
            totat_cost += cross_entropy.cost(_y, s)
            jacobian += cross_entropy.features_jacobian(_y, s, z, (self.input_size, self.feature_size))
        jacobian / x_mini_batch.size
        return totat_cost, jacobian

    def train(self, epoch, eta, mini_batch_size, data):
        for e in range(epoch):
            np.random.shuffle(data)
            mini_batch = data[:mini_batch_size]
            x_mini_batch = data[:, :self.input_size]
            y_mini_batch = data[:, self.input_size:]

            cost, jacobian = self.feed_forward(x_mini_batch, y_mini_batch)
            self.features -= eta * jacobian
            print cost, "\n"


n = Network(3, 4)
# n.forward(np.array([0,1,0]))
# n.feed_forward(np.array([(0,1,1),(0,1,1),(0,1,1)]), np.array([(1,1,0),(1,0,1),(0,1,1)]))
data = np.array([(0, 1, 1, 0, 1, 0), (0, 0, 1, 0, 0, 1), (1, 1, 1, 0, 1, 1)])
n.train(20000, 0.3, 3, data)
