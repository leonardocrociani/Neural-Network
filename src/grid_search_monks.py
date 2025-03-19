"""
This is the script used to conduct the grid search for each Monk task.
"""

from lib.data_loader import get_monks_dataset
from lib.neural_network import NeuralNetwork
from lib.grid_search import grid_search, get_best_results
from lib.utils import parse_monks_id, check_param_grid, monks_regularization
import numpy as np
np.random.seed(420)


MONKS_ID = parse_monks_id() # parse arg from cli
EPOCHS = 500
BATCH_SIZE = 'full'
ERROR_FUNCTION = "mse"

X_dev, y_dev, _, _ = get_monks_dataset(MONKS_ID, one_hot_encode=True)

param_grid_1 = {
    "learning_rate": np.arange(0.1, 0.9, 0.01),
    "layers": [ [17, i, 1] for i in [3, 4, 5] ],
    "decay_rate": [0],
    "lambda_reg": [0],
    "momentum_alpha": [0] + np.arange(0.5, 0.9, 0.1).tolist(),
    "momentum_type": ["none", "momentum"],
    "weight_init": ["glorot", "base", "he"],
    "activation_function_names": [
        [i, "sigmoid"] for i in ["relu", "tanh", "sigmoid"]
    ],
    "reg_type": ["none"],
    "lr_decay_type": ["none"],
}

param_grid_reg = param_grid_1.copy()
param_grid_reg["lambda_reg"] = [1/10**i for i in (np.arange(1,5,1).tolist())]
param_grid_reg["reg_type"] = ["l1", "l2"]
param_grid_reg['layers'] = [ [17, i, 1] for i in [4, 5] ]  
param_grid_3 = param_grid_1.copy()
param_grid_3['layers'] = param_grid_reg['layers']

params_grid = {
    '1' : param_grid_1, 
    '2' : param_grid_1, 
    '3' : param_grid_3, 
    '3_reg' : param_grid_reg
}

param_grid = params_grid[str(MONKS_ID)]

is_regularization_enabled = monks_regularization()
if is_regularization_enabled:
    param_grid = params_grid[f"{MONKS_ID}_reg"]

check_param_grid(param_grid)

# needed in gs
def model_builder(
    learning_rate,
    layers,
    decay_rate,
    lambda_reg,
    momentum_alpha,
    reg_type,
    lr_decay_type,
    activation_function_names,
    weight_init,
    momentum_type,
):
    return NeuralNetwork(
        layers=layers,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        lr_decay_type=lr_decay_type,
        activation_function_names=activation_function_names,
        weight_init=weight_init,
        momentum_type=momentum_type,
        reg_type=reg_type,
        lambda_reg=lambda_reg,
        momentum_alpha=momentum_alpha,
        error_function_name=ERROR_FUNCTION,
        task="classification",
    )

_, all_results = grid_search(
    model_builder=model_builder,
    param_grid=param_grid,
    X=X_dev,
    y=y_dev,
    k=5,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=False,
    early_stopping=False if not is_regularization_enabled else True,
    patience=20, # ignored when early stopping is False
    n_jobs=-1,
    results_folder_name=f"monks_{MONKS_ID}{'_reg' if is_regularization_enabled else ''}",
    evaluation_function_name='mse'
)

best_results = get_best_results(all_results, strategy='max_accuracy_min_metric_std')
print(best_results)