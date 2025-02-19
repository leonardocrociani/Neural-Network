from lib.data_loader import get_monks_dataset
from lib.model_selection import grid_search
from lib.utils import build_nn_model_with_params
from sklearn.model_selection import train_test_split

X_dev, y_dev, X_test, y_test = get_monks_dataset(1, one_hot_encode=True)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)


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
    epochs=500,       # Reduced epochs for quicker grid search.
    batch_size=32,
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    n_jobs=-1,        # Utilize all available processors.
    maximize=True,    # For classification, higher accuracy is better.
    verbose=True
)

print("\nFinal Grid Search Best Result:")
print(best_params)

best_params = best_params['params']

# Build a new model with the best hyperparameters.
nn_best = build_nn_model_with_params(
    learning_rate=best_params['learning_rate'],
    lr_decay_type=best_params['lr_decay_type'],
    lambda_reg=best_params['lambda_reg'],
    weight_init=best_params['weight_init']
)

X_dev, y_dev, X_test, y_test = get_monks_dataset(1, one_hot_encode=True)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

nn_best.train(
    X_train, y_train,
    epochs=1000,
    batch_size=64,
    verbose=True,
    validation_data=(X_val, y_val),
)

nn_best.plot_loss_history() # plotta curva della loss sia su train e validation

nn_best = build_nn_model_with_params(
    learning_rate=best_params['learning_rate'],
    lr_decay_type=best_params['lr_decay_type'],
    lambda_reg=best_params['lambda_reg'],
    weight_init=best_params['weight_init']
)

# now, we have to retrain the model with the best hyperparameters on the entire development set
nn_best.train(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    verbose=True
)

# Evaluate the best model on the test set.
test_accuracy = nn_best.evaluate(X_test, y_test)

print(f"\nBest Model Test Accuracy: {test_accuracy:.4f}")

nn_best.plot_loss_history()