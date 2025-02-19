import numpy as np
from sklearn.model_selection import train_test_split
from lib.data_loader import get_monks_dataset  # Assuming this module already exists.
from lib.neural_network import NeuralNetwork
from lib.cross_validation import k_fold_cross_validation
from lib.grid_search import grid_search

def main():
    # --- Example: Training on Monk's Dataset ---
    X_train, y_train, X_test, y_test = get_monks_dataset(3, one_hot_encode=True)
    input_size = X_train.shape[1]
    hidden_units = 10
    output_size = 1
    layers = [input_size, hidden_units, output_size]
    activation_funcs = ["relu", "sigmoid"]

    nn_clf = NeuralNetwork(
        layers=layers,
        learning_rate=0.2,
        lambda_reg=0.001,
        reg_type="l2",
        loss_function_name="mse",  # Using MSE for illustration.
        activation_function_names=activation_funcs,
        task="classification",
        lr_decay_type="linear",
        decay_rate=0.001,
        weight_init="base"
    )

    nn_clf.train(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        verbose=True,
        early_stopping=True,
        validation_data=(X_test, y_test),
        patience=10,
        min_delta=1e-4
    )

    accuracy = nn_clf.evaluate(X_test, y_test)
    print(f"\nNeural Network Classification Accuracy: {accuracy:.4f}")
    nn_clf.plot_loss_history()

    # --- K-fold Cross-Validation ---
    def build_nn_model():
        return NeuralNetwork(
            layers=layers,
            learning_rate=0.2,
            lambda_reg=0.001,
            reg_type="l2",
            loss_function_name="mse",
            activation_function_names=activation_funcs,
            task="classification",
            lr_decay_type="linear",
            decay_rate=0.001,
            weight_init="base"
        )

    cv_results = k_fold_cross_validation(
        model_builder=build_nn_model,
        X=X_train,
        y=y_train,
        k=5,
        epochs=1000,
        batch_size=32,
        verbose=False,
        early_stopping=True,
        patience=10,
        min_delta=1e-4
    )
    print("\nK-fold Cross-Validation Results:")
    print(cv_results)

    # --- Grid Search Example ---
    def build_nn_model_with_params(learning_rate=0.2, lambda_reg=0.001, reg_type="l2",
                                   lr_decay_type="linear", decay_rate=0.001, weight_init="base"):
        return NeuralNetwork(
            layers=layers,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            reg_type=reg_type,
            loss_function_name="mse",
            activation_function_names=activation_funcs,
            task="classification",
            lr_decay_type=lr_decay_type,
            decay_rate=decay_rate,
            weight_init=weight_init
        )

    param_grid = {
        "learning_rate": [0.1, 0.2],
        "lambda_reg": [0.001, 0.01],
        "lr_decay_type": ["linear", "none"],
        "decay_rate": [0.001, 0.0],
        "weight_init": ["base", "glorot"],
    }

    best_params, all_results = grid_search(
        model_builder=build_nn_model_with_params,
        param_grid=param_grid,
        X=X_train,
        y=y_train,
        k=5,
        epochs=500,       # Fewer epochs for quicker grid search.
        batch_size=32,
        early_stopping=True,
        patience=10,
        min_delta=1e-4,
        n_jobs=-1,
        maximize=True,
        verbose=True
    )

    print("\nFinal Grid Search Best Result:")
    print(best_params)
    best_hyperparams = best_params["params"]

    # --- Retrain Best Model on Development Set ---
    nn_best = build_nn_model_with_params(
        learning_rate=best_hyperparams['learning_rate'],
        lr_decay_type=best_hyperparams['lr_decay_type'],
        lambda_reg=best_hyperparams['lambda_reg'],
        weight_init=best_hyperparams['weight_init']
    )

    X_dev, y_dev, X_test, y_test = get_monks_dataset(1, one_hot_encode=True)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_dev, y_dev, test_size=0.2, random_state=42
    )

    nn_best.train(
        X_train_split, y_train_split,
        epochs=1000,
        batch_size=64,
        verbose=True,
        validation_data=(X_val_split, y_val_split)
    )
    nn_best.plot_loss_history()
    nn_best.plot_accuracy_history()

    # --- Retrain on Entire Development Set & Evaluate ---
    nn_best = build_nn_model_with_params(
        learning_rate=best_hyperparams['learning_rate'],
        lr_decay_type=best_hyperparams['lr_decay_type'],
        lambda_reg=best_hyperparams['lambda_reg'],
        weight_init=best_hyperparams['weight_init']
    )
    nn_best.train(
        X_dev, y_dev,
        epochs=1000,
        batch_size=32,
        verbose=True
    )

    test_accuracy = nn_best.evaluate(X_test, y_test)
    print(f"\nBest Model Test Accuracy: {test_accuracy:.4f}")
    nn_best.plot_loss_history()
    nn_best.plot_accuracy_history()

if __name__ == '__main__':
    main()
