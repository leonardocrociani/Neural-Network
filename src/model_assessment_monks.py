"""
This script has been developed to assess the best model found for the monks dataset.
First plots the curves using the dev set (splitted in training and validation), 
Then, after retraining on the whole dev set, it evaluates the model on the test set.
"""

from sklearn.model_selection import train_test_split
from lib.data_loader import get_monks_dataset
from lib.neural_network import NeuralNetwork
from lib.grid_search import get_best_results
from lib.utils import parse_monks_id, monks_regularization
import json
import numpy as np

np.random.seed(42)

MONKS_ID = parse_monks_id()
EPOCHS = 500
BATCH_SIZE = "full"
ERROR_FUNCTION = "mse"

X_dev, y_dev, X_test, y_test = get_monks_dataset(MONKS_ID, one_hot_encode=True)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

is_regularization_enabled = monks_regularization()

with open(f"../results/hyperparams_search/monks_{MONKS_ID}{'_reg' if is_regularization_enabled else ''}/gs_results.json", "r") as f:
    params = json.load(f)

params = get_best_results(params, strategy="min_mean_std")
print(params)
best_params = params["params"]

get_model = lambda: NeuralNetwork(
    learning_rate=best_params["learning_rate"],
    layers=best_params["layers"],
    decay_rate=best_params["decay_rate"],
    momentum_alpha=best_params["momentum_alpha"],
    reg_type=best_params["reg_type"],
    lambda_reg=best_params["lambda_reg"],
    activation_function_names=best_params["activation_function_names"],
    weight_init=best_params["weight_init"],
    momentum_type=best_params["momentum_type"],
    lr_decay_type=best_params["lr_decay_type"],
    error_function_name=ERROR_FUNCTION,
    task="classification",
)

nn = get_model()
nn.summary() # just print the arch and the hyperparams.

ese = nn.train(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping=False,
    verbose=False,
    validation_data=(X_val, y_val),
)
tr_acc = nn.evaluate(X_train, y_train)
tr_error = nn.train_error_history[-1]
val_error = nn.val_error_history[-1]
validation_accuracy = nn.evaluate(X_val, y_val)

print(f"Train Accuracy: {tr_acc}")
print(f"Validation Accuracy: {validation_accuracy}")
print(f"Train Error: {tr_error}")
print(f"Validation Error: {val_error}")

nn.plot_learning_curve(
    save_path=f"../results/hyperparams_search/monks_{MONKS_ID}{'_reg' if is_regularization_enabled else ''}/loss.png"
)
nn.plot_accuracy_history(
    save_path=f"../results/hyperparams_search/monks_{MONKS_ID}{'_reg' if is_regularization_enabled else ''}/accuracy.png"
)

# At this point, we can retrain on the full dev set to assess the model on the test set.

print('\n# ==== RETRAINING PHASE ==== #\n')

nn = get_model()
nn.train(
    X_dev,
    y_dev,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=False,
    early_stopping=False,
)

accuracy = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
