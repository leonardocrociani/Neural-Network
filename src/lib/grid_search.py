import numpy as np
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from lib.cross_validation import k_fold_cross_validation

def grid_search(model_builder, param_grid, X, y, k=5, epochs=1000, batch_size=32,
                early_stopping=False, patience=10, min_delta=1e-4, n_jobs=-1,
                maximize=True, verbose=True):
    """
    Searches over hyperparameters defined in param_grid.
    For each parameter combination, a model is built and evaluated using k-fold cross-validation.
    """
    grid = list(ParameterGrid(param_grid))
    results = []

    def evaluate_params(params):
        def builder():
            return model_builder(**params)
        cv_result = k_fold_cross_validation(builder, X, y, k=k, epochs=epochs,
                                              batch_size=batch_size,
                                              early_stopping=early_stopping,
                                              patience=patience,
                                              min_delta=min_delta,
                                              verbose=False)
        metric = cv_result["average_metric"]
        if verbose:
            print(f"Params: {params} => Average Metric: {metric:.4f}")
        return (params, metric)

    evaluated_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params) for params in grid
    )

    for params, metric in evaluated_results:
        results.append({"params": params, "average_metric": metric})

    best_result = max(results, key=lambda x: x["average_metric"]) if maximize else min(results, key=lambda x: x["average_metric"])

    if verbose:
        print("\nGrid Search Results:")
        for res in results:
            print(f"Params: {res['params']}, Average Metric: {res['average_metric']:.4f}")
        print(f"\nBest Params: {best_result['params']}, Best Average Metric: {best_result['average_metric']:.4f}")

    return best_result, results
