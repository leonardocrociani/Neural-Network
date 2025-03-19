"""
This file contains the activation functions and their derivatives.
"""

import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function.
    Args:
        x: np.ndarray, the input
    Returns:
        np.ndarray, the sigmoid of the input
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z, a):
    """
    Compute the sigmoid derivative.
    Args:
        z: np.ndarray, the input before activation
        a: np.ndarray, the activated input
    Returns:
        np.ndarray, the sigmoid derivative
    """
    return a * (1 - a)

def relu(x):
    """
    Compute the ReLU function.
    Args:
        x: np.ndarray, the input
    Returns:
        np.ndarray, the ReLU of the
    """
    return np.maximum(0, x)

def tanh(x):
    """
    Compute the tanh function.
    Args:
        x: np.ndarray, the input
    Returns:
        np.ndarray, the tanh of the input
    """
    return np.tanh(x)

def tanh_derivative(z, a):
    """
    Compute the tanh derivative.
    Args:
        z: np.ndarray, the input before activation
        a: np.ndarray, the activated input
    Returns:
        np.ndarray, the tanh derivative
    """
    return 1 - a**2

def relu_derivative(z, a):
    """
    Compute the ReLU derivative.
    Args:
        z: np.ndarray, the input before activation
        a: np.ndarray, the activated input
    Returns:
        np.ndarray, the ReLU derivative
    """
    return (z > 0).astype(float)

def linear(x):
    """
    "Compute" :) the linear function.
    Args:
        x: np.ndarray, the input
    Returns:
        np.ndarray, the same input
    """
    return x

def linear_derivative(z, a):
    """
    Compute the linear derivative.
    Args:
        z: np.ndarray, the input before activation
        a: np.ndarray, the activated input
    Returns:
        np.ndarray, the linear derivative
    """
    return np.ones_like(z)

activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "linear": linear, 
}

activation_derivatives = {
    "sigmoid": lambda z, a: sigmoid_derivative(z, a),
    "relu": lambda z, a: relu_derivative(z, a),
    "tanh": lambda z, a: tanh_derivative(z, a),
    "linear": lambda z, a: linear_derivative(z, a)
}
