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
def features_jacobian(y, s, z, features_shape):
    jacobian = np.zeros((features_shape[0],features_shape[1]))

    for n in range(features_shape[0]):
        for m in range(features_shape[1]):
            jacobian[n][m] = gradient(y,s,z,(n,m))
    return jacobian
# calculates the gradient of the cost with respect to the feature(weight) at the last layer[Wnxm]
# @params index--> 2dim(n*m) index of the feature being calculated
# @params y--> correct value(class) of the training data
# @params s--> predicted softmax result
# @params g--> result on the hidden layer
def gradient(y, s, z, index):
    t = 0
    for i in range(s.size):
        part_1 = -y[i]/s[i]
        part_2 = 0
        for j in range(s.size):
            if i==j and j==index[0]:
                part_2 += s[i]*(1-s[j])* z[index[1]]
            elif i!=j and j==index[0]:
                part_2 += -1*s[i]*s[j]*z[index[1]]
            else:
                part_2 += 0
        t+= part_1*part_2
    return t;

# f = (3,4)
# y = np.array([0,1,0]).reshape(3,1)
# s = np.array([0.1,0.8,0.1]).reshape(3,1)
# g = np.array([0.1,0.2,1,0]).reshape(4,1)
# mat = features_jacobian(f,y,s,g)
# print mat