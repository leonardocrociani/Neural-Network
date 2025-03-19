"""
This is the script used to conduct the grid searches (both coarse and fine) for the CUP dataset.
"""

from sklearn.model_selection import train_test_split
from lib.data_loader import get_cup_dataset
from lib.neural_network import NeuralNetwork
from lib.grid_search import grid_search, sort_gs_results
from lib.utils import parse_gs_type, check_param_grid
import numpy as np
import json
np.random.seed(420)

COARSE_OR_FINE = parse_gs_type()
EPOCHS = 500
BATCH_SIZE = [16, 32, 64] if COARSE_OR_FINE == "coarse" else 16
ERROR_FUNCTION = "mse"

X_dev, y_dev, _, _ = get_cup_dataset()  # test omitted => ts is used just for model assessment
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=42
)

coarse_param_grid = {
    # 0 for no regularization => (when reg_type is none), the other cases will be skipped.
    # 0 for no decay => (when lr_decay_type is none), the other cases will be skipped.
    # 0 for no momentum => (when momentum_type is none), the other cases will be skipped.
    "learning_rate": [0.001, 0.0001, 0.00001],
    "layers": [[12, 10, 3], [12, 15, 3], [12, 20, 3]],
    "decay_rate": [0, 1e-1, 1e-2, 1e-04, 1e-06],
    "lambda_reg": [0, 1e-7, 1e-5, 1e-3, 1e-1],
    "momentum_alpha": [0, 0.8, 0.7, 0.6],
    "momentum_type": ["none", "momentum"],
    "weight_init": ["glorot", "base", "he"],
    "activation_function_names": [
        ["relu", "linear"],
        ["tanh", "linear"],
    ],
    "reg_type": ["none", "l1", "l2"],
    "lr_decay_type": ["none", "linear", "exponential"],
}

coarse_best_params = {}
if COARSE_OR_FINE == "fine":
    print("🧰 Loading best params from coarse grid search...")
    with open("../results/hyperparams_search/cup/coarse/gs_results.json", "r") as f:
        params = json.load(f)
    sorted_params = sort_gs_results(params)
    coarse_best_params = sorted_params[0]["params"]

fine_param_grid = {
    "learning_rate": np.arange(0.0001, 0.001, 0.00005),
    "layers": [
        [12, 20, 3],
    ],
    "decay_rate": [0] + np.arange(0.00001, 0.0001, 0.00001).tolist(),
    "lambda_reg": [1 / 10**i for i in (np.arange(1, 5, 1).tolist())],
    "momentum_alpha": [0.75, 0.8, 0.82],
    "momentum_type": ["momentum"],
    "weight_init": ["glorot", "he"],
    "activation_function_names": [["relu", "linear"]],
    "reg_type": ["l2"],
    "lr_decay_type": ["none", "linear"],
}


param_grid = coarse_param_grid if COARSE_OR_FINE == "coarse" else fine_param_grid
check_param_grid(param_grid)

# this is required by the grid search.
def model_builder(
    learning_rate=coarse_best_params.get("learning_rate"),
    layers=coarse_best_params.get("layers"),
    decay_rate=coarse_best_params.get("decay_rate"),
    lambda_reg=coarse_best_params.get("lambda_reg"),
    momentum_alpha=coarse_best_params.get("momentum_alpha"),
    reg_type=coarse_best_params.get("reg_type"),
    lr_decay_type=coarse_best_params.get("lr_decay_type"),
    activation_function_names=coarse_best_params.get("activation_function_names"),
    weight_init=coarse_best_params.get("weight_init"),
    momentum_type=coarse_best_params.get("momentum_type"),
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
        task="regression",
    )


best_params, all_results = grid_search(
    model_builder=model_builder,
    param_grid=param_grid,
    X=X_dev,
    y=y_dev,
    k=5,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=False,
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    n_jobs=-1,
    evaluation_function_name="mee", # the evaluation metric on the folds (5)
    results_folder_name=f"cup/{COARSE_OR_FINE}",
)

print("\nFinal Grid Search Best Result:")
print(best_params)
