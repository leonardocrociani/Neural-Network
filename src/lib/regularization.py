import numpy as np

def compute_reg_gradient(W, lambda_reg, reg_type, m):
    if reg_type == "l2":
        return lambda_reg * W / m
    elif reg_type == "l1":
        return lambda_reg * np.sign(W) / m
    else:
        return 0

def compute_reg_loss(W_list, lambda_reg, reg_type):
    if reg_type == "l2":
        return (lambda_reg / 2) * sum(np.sum(W ** 2) for W in W_list)
    elif reg_type == "l1":
        return lambda_reg * sum(np.sum(np.abs(W)) for W in W_list)
    else:
        return 0
