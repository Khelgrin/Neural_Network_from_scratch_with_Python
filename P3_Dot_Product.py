import numpy as np

inputs = [1, 2, 3, 2.5]

"""
Wiemy, ze mamy 3 neurony przez to, że mamy 3 zestawy wag
"""
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
"""
Chcemy, żeby nasze wyjście było wyjściem z 3 neuronów, dlatego dajemy weights jako 1 argument, bo chcemy, żeby nasz output
był indeksowany po zestawach wag
"""
output = np.dot(weights, inputs) + biases
"""
pod spodem wyglada to tak:  
np.dot(weights, inputs) = [np.dot(weights[0], inputs),  np.dot(weights[1], inputs), np.dot(weights[2], inputs)]
"""
print(output)


"""
Dla 1 neuronu
"""
# import numpy as np

# inputs = [1, 2, 3, 2.5]
#
# weights = [0.2, 0.8, -0.5, 1.0]
#
# bias = 2
# output = np.dot(weights, inputs) + bias
# #np.dot mnoży każdy element tablicy przez każdy odpowiedni z 2 tablicy i je sumuje
# # weights[0]*inputs[0] + weights[1]*inputs[1] + weights[2]*inputs[2]
# print(output)

"""
RAW PYTHON
"""
# layer_output = []

# for neuron_weights, neuron_bias in zip(weights, biases):
#     print("weights: ", neuron_weights)
#     print("neuron bias: ", neuron_bias)
#     print('+==================+')
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         print(n_input, end="")
#         print('*', end="")
#         print(weight, end="")
#         neuron_output += n_input*weight
#         print('+', end="")
#     print('')
#     print("neuron output: ", neuron_output, '\n')
#     layer_output.append(neuron_output)
# print(layer_output)