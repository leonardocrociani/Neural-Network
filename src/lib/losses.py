import numpy as np

def binary_crossentropy_loss(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

def mee_loss(y_true, y_pred):
    diff = y_true - y_pred
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(dist)

def mee_derivative(y_true, y_pred):
    diff = y_pred - y_true
    dist = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True))
    epsilon = 1e-8
    dist_safe = np.where(dist == 0, epsilon, dist)
    N = y_true.shape[0]
    return diff / dist_safe / N

loss_functions = {
    "binary_crossentropy": binary_crossentropy_loss,
    "mse": mse_loss,
    "mee": mee_loss,
}

loss_derivatives = {
    "binary_crossentropy": binary_crossentropy_derivative,
    "mse": mse_derivative,
    "mee": mee_derivative,
}
