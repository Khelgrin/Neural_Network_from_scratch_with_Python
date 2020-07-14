import numpy as np

np.random.seed(0)

# The sample has 4 characteristics, and we are providing a batch of 3 samples

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # kształt macierzy jest odwrotny (był 3x4 teraz jest 4x3), żeby nie trzeba było robić transpozycji przekazując wynik do przodu sieci
        # wagi są losową macierzą
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)


input_parameters = len(X[0])  #number of characteristics of an object that is investigated
desired_neurons_in_layer_1 = 5
desired_neurons_in_layer_2 = 2
layer1 = LayerDense(input_parameters, desired_neurons_in_layer_1)
layer2 = LayerDense(desired_neurons_in_layer_1, desired_neurons_in_layer_2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)