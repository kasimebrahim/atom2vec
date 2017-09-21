"""
9/18/17
kasim 
se.kasim.ebrahim@gmail.com
"""

import numpy as np


def cost(y, s):
    return -np.sum(_y * np.log(_s) for _y, _s in zip(y, s))


# returns the jacobian matrix of the features
# @params features_shape size of the feature matrix
def weights_two_jacobian(y, s, z, features_shape):
    jacobian = np.zeros((features_shape[0], features_shape[1]))

    for n in range(features_shape[0]):
        for m in range(features_shape[1]):
            jacobian[n][m] = feature_gradient(y, s, z, (n, m))
    return jacobian


def weights_one_jacobian(y, s, f, _input, weights_shape):
    jacobian = np.zeros((weights_shape[0], weights_shape[1]))

    for m in range(weights_shape[0]):
        for n in range(weights_shape[1]):
            jacobian[m][n] = weight_gradient(y, s, _input, f, (m, n))
    return jacobian

# calculates the gradient of the cost with respect to the feature(weight) at the last layer[Wnxm]
# @params index--> 2dim(n*m) index of the feature being calculated
# @params y--> correct value(class) of the training data
# @params s--> predicted softmax result
# @params g--> result on the hidden layer
def feature_gradient(y, s, z, index):
    t = 0
    for i in range(s.size):
        part_1 = -y[i] / s[i]
        part_2 = 0
        for j in range(s.size):
            if i == j and j == index[0]:
                part_2 += s[i] * (1 - s[j]) * z[index[1]]
            elif i != j and j == index[0]:
                part_2 += -1 * s[i] * s[j] * z[index[1]]
            else:
                part_2 += 0
        t += part_1 * part_2
    return t;


def weight_gradient(y, s, _input, f, index):
    t = 0
    for i in range(s.size):
        part_1 = -y[i] / s[i]
        part_2 = 0
        for j in range(s.size):
            part_3 = 0;
            if i == j and j == index[0]:
                part_3 = s[i] * (1 - s[j])
            elif i != j and j == index[0]:
                part_3 = -1 * s[i] * s[j]
            else:
                part_3 = 0
            part_4 = 0
            for k in range(index[0]):
                part_4 += f[j][k] * _input[index[1]]
            part_2+=part_3*part_4
        t += part_1 * part_2
    return t;
