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

        self.weights_one = np.random.randn(self.feature_size, self.input_size)
        self.weights_two = np.random.randn(self.input_size, self.feature_size)

        self.features = np.zeros((self.input_size, self.feature_size))

    def forward(self, _input):
        z = np.dot(self.weights_one, _input)
        self.features[np.argmax(_input)] = z.reshape(z.size);
        g = np.dot(self.weights_two, z)
        s = softmax.func(g)
        return z, g, s

    def feed_forward(self, x_mini_batch, y_mini_batch):
        weights_two_jacobian = np.zeros((self.input_size, self.feature_size))
        weights_one_jacobian = np.zeros((self.feature_size, self.input_size))

        totat_cost = 0
        counter = 0
        j=0
        for x, y in zip(x_mini_batch, y_mini_batch):
            _x = x.reshape(x.size, 1)
            _y = y.reshape(y.size, 1)

            z, g, s = self.forward(_x)
            if s.argmax() == _y.argmax():
                counter += 1
            j+=1

            totat_cost += cross_entropy.cost(_y, s)
            weights_two_jacobian += cross_entropy.weights_two_jacobian(_y, s, z, (self.input_size, self.feature_size))
            weights_one_jacobian += cross_entropy.weights_one_jacobian(_y, s, self.weights_two, _x,
                                                               (self.feature_size, self.input_size))

        weights_two_jacobian = weights_two_jacobian / x_mini_batch.size
        weights_one_jacobian = weights_one_jacobian / x_mini_batch.size
        return counter, totat_cost, weights_two_jacobian, weights_one_jacobian

    def train(self, epoch, eta, mini_batch_size, data):
        initial_cost = 0;
        for e in range(epoch):
            np.random.shuffle(data)
            mini_batch = data[:mini_batch_size]
            x_mini_batch = mini_batch[:, :self.input_size]
            y_mini_batch = mini_batch[:, self.input_size:]

            counter, cost, weights_two_jacobian, weights_one_jacobian = self.feed_forward(x_mini_batch, y_mini_batch)
            self.weights_two -= eta * weights_two_jacobian
            self.weights_one -= eta * weights_one_jacobian
            if e == 0:
                initial_cost = cost
            print "epoch ", e, " cost = ", cost, " correct = ", counter,"\n"
        print"initial cost was :", initial_cost, "\n"
        # print self.features
        return self.features
