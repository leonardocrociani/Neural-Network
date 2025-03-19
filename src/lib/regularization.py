"""
Compute regularizaiton loss and gradient
"""

import numpy as np

def compute_reg_loss(W_list, lambda_reg, reg_type):
    """
    Computes the regularization loss for a given list of weight matrices W_list.
    Not applied to biases, as indicated in the slides.
    Args:
        W_list: list of np.ndarray, the weight matrices
        lambda_reg: float, the regularization parameter
        reg_type: str, the regularization type
    Returns:
        float, the regularization loss
    """
    if reg_type == "l2":
        return (lambda_reg / 2) * sum(np.sum(W ** 2) for W in W_list) # /2 to "simplify" the gradient (below)
    elif reg_type == "l1":
        return lambda_reg * sum(np.sum(np.abs(W)) for W in W_list)
    else:
        return 0


def compute_reg_gradient(W, lambda_reg, reg_type):
    """
    Computes the regularization gradient for a given weight matrix W.
    [NOTE: The value will be divided by the #samples in the _compute_gradients method]
    Args:
        W: np.ndarray, the weight matrix
        lambda_reg: float, the regularization parameter
        reg_type: str, the regularization type
    Returns:
        np.ndarray, the regularization
    """
    if reg_type == "l2":
        return lambda_reg * W
    elif reg_type == "l1":
        return lambda_reg * np.sign(W) 
    else:
        return 0