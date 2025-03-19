"""
[Less Relevant Code]
This script simply report the monk's task results with different trials.
"""

from sklearn.model_selection import train_test_split
from lib.data_loader import get_monks_dataset
from lib.neural_network import NeuralNetwork
from lib.grid_search import get_best_results
import json
import numpy as np
np.random.seed(42)

EPOCHS = 500
BATCH_SIZE = "full"
ERROR_FUNCTION = "mse"
TRIAL = 3

for MONKS_ID in [1, 2, 3, '3_reg']:
    
    if MONKS_ID == '3_reg':
        MONKS_ID = 3
        is_regularization_enabled = True
    else:
        is_regularization_enabled = False
    

    X_dev, y_dev, X_test, y_test = get_monks_dataset(MONKS_ID, one_hot_encode=True)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

    with open(f"../results/hyperparams_search/monks_{MONKS_ID}{'_reg' if is_regularization_enabled else ''}/gs_results.json", "r") as f:
        params = json.load(f)

    params = get_best_results(params, strategy="min_mean_std")
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

    results = {
        "train_accuracy": [],
        "train_error": [],
        "validation_error": [],
        "validation_accuracy": [],
        "test_error": [],
        "test_accuracy": [],
    }

    for i in range(TRIAL):
        nn = get_model()
        if i == 0: 
            nn.summary()
        
        nn.train(
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

        
        nn = get_model()
        nn.train(
            X_dev,
            y_dev,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=False,
            early_stopping=False,
        )

        mse = nn.train_error_history[-1]
        accuracy = nn.evaluate(X_test, y_test)
        
        results["train_accuracy"].append(tr_acc)
        results["train_error"].append(tr_error)
        results["validation_error"].append(val_error)
        results["validation_accuracy"].append(validation_accuracy)
        results["test_error"].append(mse)
        results["test_accuracy"].append(accuracy)
        
    print(f"Results for MONKS-{MONKS_ID}{'_reg' if is_regularization_enabled else ''} after {TRIAL} trials:")
    print('AVG Train Accuracy:', sum(results["train_accuracy"]) / len(results["train_accuracy"]))
    print('AVG Validation Accuracy:', sum(results["validation_accuracy"]) / len(results["validation_accuracy"]))
    print('AVG Test Accuracy:', sum(results["test_accuracy"]) / len(results["test_accuracy"]))
    print('AVG Train Error:', sum(results["train_error"]) / len(results["train_error"]))
    print('AVG Validation Error:', sum(results["validation_error"]) / len(results["validation_error"]))
    print('AVG Test Error:', sum(results["test_error"]) / len(results["test_error"]))
