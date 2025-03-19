"""
Utility functions for parsing command line arguments and checking hyperparameters.
"""

import sys

def parse_gs_type():
    """
    Parse <coarse|fine> from the command line.
    """
    if len(sys.argv) != 2:
        raise ValueError("Usage: python grid_search_cup.py <coarse|fine>")

    COARSE_OR_FINE = sys.argv[1]
    print(f"✅ Working with {COARSE_OR_FINE.upper()} grid search")

    if COARSE_OR_FINE not in ["coarse", "fine"]:
        raise ValueError("Usage: python grid_search_cup.py <coarse|fine>")
    
    return COARSE_OR_FINE

def parse_monks_id():
    """
    Parse the monk ID from the command line. 
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: python grid_search_monks.py <1|2|3|3_reg>")

    MONKS_ID = int(sys.argv[1].replace('_reg', '').strip())
    print(f"✅ Working with MONKS-{MONKS_ID} dataset")

    if MONKS_ID not in [1, 2, 3]:
        raise ValueError("Usage: python grid_search_monks.py <1|2|3|3_reg>")
    
    return MONKS_ID

def monks_regularization():
    """
    Parse the regularization flag from the command line.
    """
    regularization =  len(sys.argv) >= 2 and 'reg' in sys.argv[1]
    if regularization:
        print("✅ (λ) Regularization is enabled")
    return regularization

def check_param_grid(param_grid):

    """
    Check if the parameters grid is valid or not:
    If reg_type includes none, then lambda_reg list must includes 0.
    If reg_type does not include none, then lambda_reg list must not include 0.
    If lr_decay_type includes none, then decay_rate list must includes 0.
    If lr_decay_type does not include none, then decay_rate list must not include 0.
    If momentum_type includes none, then momentum_alpha list must includes 0.
    If momentum_type does not include none, then momentum_alpha list must not include 0..
    """
    
    if "reg_type" in param_grid:
        if "none" in param_grid["reg_type"]:
            if 0 not in param_grid["lambda_reg"]:
                raise ValueError("When reg_type is none, you MUST include lambda_reg equal to 0.")
        else:
            if 0 in param_grid["lambda_reg"]:
                raise ValueError("When reg_type is not none, you CANNOT include lambda_reg equal to 0.")
            
    if "lr_decay_type" in param_grid:
        if "none" in param_grid["lr_decay_type"]:
            if 0 not in param_grid["decay_rate"]:
                raise ValueError("When lr_decay_type is none, you MUST include decay_rate equal to 0.")
        else:
            if 0 in param_grid["decay_rate"]:
                raise ValueError("When lr_decay_type is not none, you CANNOT include decay_rate equal to 0.")
            
    if "momentum_type" in param_grid:
        if "none" in param_grid["momentum_type"]:
            if 0 not in param_grid["momentum_alpha"]:
                raise ValueError("When momentum_type is none, you MUST include momentum_alpha equal to 0.")
        else:
            if 0 in param_grid["momentum_alpha"]:
                raise ValueError("When momentum_type is not none, you CANNOT include momentum_alpha equal to 0.")

def invalid_hyperparams_combo(nn_model):
    """
    Check if the hyperparameters are invalid or not, with the same logic as check_param_grid.
    """
    if nn_model.reg_type == "none" and nn_model.lambda_reg != 0:
        return True
    if nn_model.reg_type != "none" and nn_model.lambda_reg == 0:
        return True
    if nn_model.lr_decay_type == "none" and nn_model.decay_rate != 0:
        return True
    if nn_model.lr_decay_type != "none" and nn_model.decay_rate == 0:
        return True
    if nn_model.momentum_type == "none" and nn_model.momentum_alpha != 0:
        return True
    if nn_model.momentum_type != "none" and nn_model.momentum_alpha == 0:
        return True
    return False