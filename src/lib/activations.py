import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z, a):
    return a * (1 - a)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(z, a):
    return (z > 0).astype(float)

def linear(x):
    return x

def linear_derivative(z, a):
    return np.ones_like(a)

activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear
}

activation_derivatives = {
    "sigmoid": lambda z, a: sigmoid_derivative(z, a),
    "relu": lambda z, a: relu_derivative(z, a),
    "linear": lambda z, a: linear_derivative(z, a)
}
