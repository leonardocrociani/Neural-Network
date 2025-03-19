"""
This code implements a grid_search (parallelized).
Furthermore it provides functions for choosing the best results from the grid search.
"""

import numpy as np
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from lib.cross_validation import k_fold_cross_validation
import json
from tqdm import tqdm
np.random.seed(420)

def grid_search(
    model_builder,
    param_grid,
    X,
    y,
    k=5,
    epochs=1000,
    batch_size=32,  # oppure una lista di batch size, oppure 'full' per usare tutto il dataset
    early_stopping=False,
    patience=10,
    min_delta=1e-4,
    n_jobs=-1,
    verbose=True,
    results_folder_name=None,
    evaluation_function_name=None,
):
    """
    Perform a grid search on the given model_builder and param_grid.
    Args:
        model_builder: a function that returns a model.
        param_grid: a dictionary containing the hyperparameters to search.
        X: the input data.
        y: the target data.
        k: the number of folds for the cross validation.
        epochs: the number of epochs for the training.
        batch_size: the batch size for the training.
        early_stopping: whether to use early stopping.
        patience: the patience for the early stopping.
        min_delta: the minimum delta for the early stopping.
        n_jobs: the number of jobs to run in parallel.
        verbose: whether to print the results.
        results_folder_name: the name of the folder where to save the results.
        evaluation_function_name: the name of the evaluation function to use.
    Returns:
        the best result and all the results.
    """
    grid = list(ParameterGrid(param_grid))
    
    # if batch_size is a list, each value is embedded in the parameters grid
    if isinstance(batch_size, list):
        new_grid = []
        for params in grid:
            for bs in batch_size:
                new_params = params.copy()
                new_params["batch_size"] = bs
                new_grid.append(new_params)
        grid = new_grid

    print('Total combinations: ', len(grid))
    results = []

    def evaluate_params(params):
        # extract the batch_size from the parameters if available, otherwise use the provided value
        actual_batch_size = params.pop("batch_size", batch_size)

        def builder():
            # model_builder should not receive the batch_size parameter.
            return model_builder(**params)
        
        cv_result = k_fold_cross_validation(
            builder,
            X,
            y,
            k=k,
            epochs=epochs,
            batch_size=actual_batch_size if actual_batch_size != "full" else X.shape[0],
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            verbose=False,
            evaluation_function_name=evaluation_function_name,
        )
        
        metric = cv_result["average_metric"]
        std_metric = cv_result["std_metric"]
        accuracy = cv_result["accuracy"] if "accuracy" in cv_result else None
        if verbose:
            print(f"Params: {params} with batch_size: {actual_batch_size} => Average Metric: {metric:.4f}, Std dev: {std_metric:.4f}")
       
        evaluation_result = {
            "params": params,
            "batch_size": actual_batch_size,
            "average_metric": metric,
            "std_metric": std_metric,
        }
        
        if accuracy is not None:
            evaluation_result["accuracy"] = accuracy
            evaluation_result["std_accuracy"] = cv_result["std_accuracy"]
        
        return evaluation_result

    evaluated_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params.copy()) for params in tqdm(grid, desc="Grid Search Progress")
    )

    for eval_result in evaluated_results:
        # restore the batch_size in the parameters
        params = eval_result["params"]
        params["batch_size"] = eval_result.get("batch_size", batch_size)
        metric = eval_result["average_metric"]
        std_metric = eval_result["std_metric"]
        accuracy = eval_result["accuracy"] if "accuracy" in eval_result else None
        std_accuracy = eval_result["std_accuracy"] if "std_accuracy" in eval_result else None
        
        result = {"params": params, "average_metric": metric, "std_metric" : std_metric}
        
        if accuracy is not None:
            result["accuracy"] = accuracy
            result["std_accuracy"] = std_accuracy
            
        results.append(result)

    best_result = get_best_results(results)

    if verbose or results_folder_name is not None:
        if verbose:
            print("\nGrid Search Results:")
            
        if results_folder_name is not None:
            filepath = f"../results/hyperparams_search/{results_folder_name}/gs_results.json"
            with open(filepath, "w") as file:
                json.dump(results, file)
            print(f"Results saved to {filepath}")

        if verbose:
            for res in results:
                print(f"Params: {res['params']}, Average Metric: {res['average_metric']:.4f}")
            print(f"\nBest Params: {best_result['params']}, Best Average Metric: {best_result['average_metric']:.4f}")

    return best_result, results


def get_best_results(results, strategy='min_mean_std'):
    """
    Get the best results from the grid search.
    Args:
        results: the results of the grid search.
        strategy: the strategy to use for selecting the best results, among: 'min_mean_std', 'first_mixed_position_mean_std'.
    Returns:
        the best result.
    """
    if strategy == 'min_mean_std':    
        print('Using SELECTION STRATEGY min_mean_std')
        return min(
            results, 
            key=lambda x: (round(x["average_metric"], 4), x["std_metric"])
        )
        
    elif strategy == 'min_std_mean':
        print('Using SELECTION STRATEGY min_std_mean')
        return min(
            results, 
            key=lambda x: (round(x["std_metric"], 4), x["average_metric"])
        )
        
    elif strategy == 'first_mixed_position_mean_std':
        print('Using SELECTION STRATEGY first_mixed_position_mean_std')
        sorted_by_mean = sorted(results, key=lambda x: x["average_metric"])
        mean_ranks = {id(el): rank for rank, el in enumerate(sorted_by_mean)}
        
        sorted_by_std = sorted(results, key=lambda x: x["std_metric"])
        std_ranks = {id(el): rank for rank, el in enumerate(sorted_by_std)}
        
        best_element = min(results, key=lambda x: mean_ranks[id(x)] + std_ranks[id(x)])
        
        return best_element
    
    elif strategy == 'max_accuracy_min_metric_std':
        print('Using SELECTION STRATEGY max_accuracy_min_metric_std')
        sorted_by_acc = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
        return min(
            sorted_by_acc, 
            key=lambda x: (round(x["average_metric"], 4), x["std_metric"])
        )
        
    else:
        raise ValueError("Invalid strategy. Choose between 'min_mean_std' and 'first_mixed_position_mean_std'")

    
def sort_gs_results(results, strategy='min_mean_std'):
    """
    Sort the results of the grid search.
    :param results: the results of the grid search.
    :param strategy: the strategy to use for sorting the results, among: 'min_mean_std', 'first_mixed_position_mean_std'.
    :return: the sorted results.
    """
    if strategy == 'min_mean_std':    
        return sorted(
            results, 
            key=lambda x: (round(x["average_metric"], 4), x["std_metric"])
        )
        
    elif strategy == 'first_mixed_position_mean_std':
        sorted_by_mean = sorted(results, key=lambda x: x["average_metric"])
        mean_ranks = {id(el): rank for rank, el in enumerate(sorted_by_mean)}
        
        sorted_by_std = sorted(results, key=lambda x: x["std_metric"])
        std_ranks = {id(el): rank for rank, el in enumerate(sorted_by_std)}
        
        return sorted(results, key=lambda x: mean_ranks[id(x)] + std_ranks[id(x)])
        
    else:
        raise ValueError("Invalid strategy. Choose between 'min_mean_std' and 'first_mixed_position_mean_std'")