"""
This code implements the error function. Below, sometime clipping is necessary because log(0) is not defined.
To avoid log(0), we replace the values of y_pred that are 0 with a very small value.
"""

import numpy as np

def binary_crossentropy_loss(y_true, y_pred):
    """
    Compute the binary crossentropy loss.
    Args:
        y_true: np.ndarray, the true labels
        y_pred: np.ndarray, the predicted labels
    Returns:
        float, the binary crossentropy loss
    """
    epsilon = 1e-8 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    """
    Compute the binary crossentropy derivative.
    [NOTE: The value will be divided by the #samples in the _compute_gradients method]
    Args:
        y_true: np.ndarray, the true labels
        y_pred: np.ndarray, the predicted labels
    Returns:
        np.ndarray, the binary crossentropy derivative
    """
    epsilon = 1e-8 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def mse_loss(y_true, y_pred):
    """
    Compute the mean squared error loss.
    Args:
        y_true: np.ndarray, the true labels
        y_pred: np.ndarray, the predicted labels
    Returns:
        float, the mean squared error loss
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    """
    Compute the mean squared error derivative.
    [NOTE: The value will be divided by the #samples in the _compute_gradients method]
    Args:
        y_true: np.ndarray, the true labels
        y_pred: np.ndarray, the predicted labels
    Returns:
        np.ndarray, the mean squared error derivative
    """
    return 2 * (y_pred - y_true) 

def mee_loss(y_true, y_pred):
    """
    Compute the mean euclidean error loss.
    Args:
        y_true: np.ndarray, the true labels
        y_pred: np.ndarray, the predicted labels
    Returns:
        float, the mean euclidean error loss
    """
    diff = y_true - y_pred
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(dist)

error_functions = {
    "binary_crossentropy": binary_crossentropy_loss,
    "mse": mse_loss,
    "mee": mee_loss,
}

error_functions_derivatives = {
    "binary_crossentropy": binary_crossentropy_derivative,
    "mse": mse_derivative, # in realtà è stata usata solo MSE
}
