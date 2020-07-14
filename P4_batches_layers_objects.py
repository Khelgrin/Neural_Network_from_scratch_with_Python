import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]
# # """My solution to the shape error"""
# layer_output = []
# for input in inputs:
#     output = np.dot(weights, input) + biases
#     layer_output.append(output)
# print(np.array(layer_output))
# my result:
# [4.8   1.21  2.385]
# [ 8.9  -1.81  0.2 ]
# [1.41  1.051 0.026]

# intended way of solving the shape error
layer1_output = np.dot(inputs, np.array(weights).T) + biases
# result:
# [[ 4.8    1.21   2.385]
#  [ 8.9   -1.81   0.2  ]
#  [ 1.41   1.051  0.026]]

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
# result:
# [[ 0.5031  -1.04185 -2.03875]
#  [ 0.2434  -2.7332  -5.7633 ]
#  [-0.99314  1.41254 -0.35655]]
print(layer2_output)