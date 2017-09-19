"""
9/18/17
kasim 
se.kasim.ebrahim@gmail.com
"""


# TEST  "this outstanding performance is achieved"
# used one hot encoding [i:e "this" = 1,0,0,0,0]
# used context size of 1 [i:e "outstanding has "this" and "performance" in its context]
import numpy as np

import Network

def get_distance_vector(matrix, i):
    distance_vector = np.zeros(5)
    for j in range(matrix.shape[0]):
        distance = 0
        for k in range(matrix.shape[1]):
            distance+=(matrix[i][k]-matrix[j][k])**2
        distance_vector[j] = distance
    return distance_vector

n = Network.Network(5, 4)
data = np.array([(1, 0, 0, 0, 0, 0, 1, 0, 0, 0), (0, 1, 0, 0, 0, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 1, 0, 0),
                 (0, 0, 1, 0, 0, 0, 1, 0, 0, 0), \
                 (0, 0, 1, 0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 1, 0, 0, 0, 1, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
                 (0, 0, 0, 0, 1, 0, 0, 0, 1, 0)])
features = n.train(2000, 0.03, 5, data)
print get_distance_vector(features, 0)
print get_distance_vector(features, 1)
print get_distance_vector(features, 2)
print get_distance_vector(features, 3)
print get_distance_vector(features, 4)