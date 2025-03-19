"""
This module contains the k-fold cross-validation function.
"""

import numpy as np
from sklearn.model_selection import KFold
from lib.utils import invalid_hyperparams_combo
from lib.neural_network import NeuralNetwork
np.random.seed(420)

def k_fold_cross_validation(
    model_builder,
    X,
    y,
    k=5,
    epochs=1000,
    batch_size=32, # or "full" to use the full dataset
    verbose=True,
    early_stopping=False,
    patience=10,
    min_delta=1e-4,
    evaluation_function_name=None
):
    """
    Perform a k-fold cross-validation on the given model_builder.
    Args:
        model_builder: a function that returns a model.
        X: the input data.
        y: the target data.
        k: the number of folds for the cross validation.
        epochs: the number of epochs for the training.
        batch_size: the batch size for the training.
        verbose: whether to print the results.
        early_stopping: whether to use early stopping.
        patience: the patience for the early stopping.
        min_delta: the minimum delta for the early stopping.
        evaluation_function_name: the name of the evaluation function to use.  
    Returns:
        a dictionary containing the metrics for each fold, the average metric and the std metric
    """
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    fold_accuracys = []
    fold = 1

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        nn_model:NeuralNetwork = model_builder()
        
        if invalid_hyperparams_combo(nn_model): # tipo reg_type = none e lambda_reg != 0, oppure lr_decay_type = none e decay_rate != 0, oppure momentum_type = none e momentum_alpha != 0
            fold_metrics.append(np.inf)
            if verbose:
                print(f"Fold {fold} Evaluation Metric: inf")
            break
        
        nn_model.train(
            X_train_fold,
            y_train_fold,
            epochs=epochs,
            batch_size=batch_size if batch_size != "full" else X_train_fold.shape[0],
            verbose=verbose,
            early_stopping=early_stopping,
            validation_data=(X_val_fold, y_val_fold),
            patience=patience,
            min_delta=min_delta,
        )

        metric = nn_model.evaluate(X_val_fold, y_val_fold, evaluation_function_name=evaluation_function_name, discretize=False)
        
        if nn_model.task == 'classification':
            accuracy = nn_model.evaluate(X_val_fold, y_val_fold, discretize=True)
            fold_accuracys.append(accuracy)

        if verbose:
            print(f"Fold {fold} Evaluation Metric: {metric:.4f}")
        fold_metrics.append(metric)
        fold += 1

    avg_metric = np.mean(fold_metrics)
    std_metric = np.inf if np.isinf(fold_metrics[0]) else np.std(fold_metrics)
    
    kfold_result = {"fold_metrics": fold_metrics, "average_metric": avg_metric, "std_metric": std_metric}
    
    if len(fold_accuracys) > 0:
        avg_accuracy = np.mean(fold_accuracys)
        std_accuracy = np.std(fold_accuracys)
        kfold_result["accuracy"] = avg_accuracy
        kfold_result["std_accuracy"] = std_accuracy
    
    if verbose:
        print(f"\nAverage Evaluation Metric over {k} folds: {avg_metric:.4f} +/- {std_metric:.4f}")
        
    return kfold_result
