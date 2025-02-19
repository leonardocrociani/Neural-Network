import numpy as np
from sklearn.model_selection import KFold

def k_fold_cross_validation(model_builder, X, y, k=5, epochs=1000, batch_size=32,
                            verbose=True, early_stopping=False, patience=10, min_delta=1e-4):
    """
    Performs k-fold cross validation by splitting the data, creating a new model 
    for each fold via the provided model_builder, training it, and evaluating its metric.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold}/{k}")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        nn_model = model_builder()
        nn_model.train(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            early_stopping=early_stopping,
            validation_data=(X_val_fold, y_val_fold),
            patience=patience,
            min_delta=min_delta
        )

        metric = nn_model.evaluate(X_val_fold, y_val_fold)
        print(f"Fold {fold} Evaluation Metric: {metric:.4f}")
        fold_metrics.append(metric)
        fold += 1

    avg_metric = np.mean(fold_metrics)
    print(f"\nAverage Evaluation Metric over {k} folds: {avg_metric:.4f}")
    return {"fold_metrics": fold_metrics, "average_metric": avg_metric}
