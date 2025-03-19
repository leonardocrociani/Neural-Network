"""
This script is the one used to assess the final model (identified by the best hyperparameters) on the CUP dataset.
First plots the curves using the dev set (splitted in training and validation), 
Then, after retraining on the whole dev set, it evaluates the model on the test set.
Lastly, it retrain on the full dataset and save the predictions on the final test set.
"""


from sklearn.model_selection import train_test_split
from lib.data_loader import get_cup_dataset, get_cup_final_test_set, save_final_prediction
from lib.grid_search import get_best_results
from lib.neural_network import NeuralNetwork
import json
import numpy as np
np.random.seed(420)

SAVE_FINAL_PREDICTIONS = True
EPOCHS = 500
ERROR_FUNCTION = 'mse'

X_dev, y_dev, X_test, y_test = get_cup_dataset()
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=42
)

# loading the gs results and getting the best hyperparams.
with open("../results/hyperparams_search/cup/fine/gs_results.json", "r") as f:
    results = json.load(f)

best_params = get_best_results(results)['params']
BATCH_SIZE = best_params['batch_size']

print('Batch size:', BATCH_SIZE)

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
    task="regression",
)

nn = get_model()
nn.summary() # just print the arch and hyperparams.

ese = nn.train(
    X_train_split,
    y_train_split,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping=True,
    patience=10,
    min_delta=1e-4,
    verbose=False,
    validation_data=(X_val_split, y_val_split),
)

nn.plot_learning_curve(save_path="../results/hyperparams_search/cup/fine/loss_dev.png")
nn.plot_learning_curve(save_path="../results/hyperparams_search/cup/fine/loss_dev_zoomed.png", zoomed=True)

mee_train = nn.evaluate(X_train_split, y_train_split, evaluation_function_name="mee")
mee_validation = nn.evaluate(X_val_split, y_val_split, evaluation_function_name="mee")

print('MEE on the training set:', mee_train)
print('MEE on the validation set:', mee_validation)


# At this point, we can retrain on the full dev set to assess the model on the internal Test Set.

print('\n# ==== RETRAINING PHASE ==== #\n')

nn = get_model() # get a brand new instance of the NN
final_epochs = ese if ese is not None else EPOCHS # if the early stopping was triggered, we use the epoch where it stopped

nn.train(
    X_dev,
    y_dev,
    epochs=final_epochs,
    batch_size=BATCH_SIZE,
    verbose=False,
)

mee_test = nn.evaluate(X_test, y_test, evaluation_function_name="mee")
print(f"Final MEE on the test set: {mee_test}")

if SAVE_FINAL_PREDICTIONS:
    
    # For the blind test, we retrain on the full dataset and
    # save the predictions on the final test set.
    
    X_full, y_full, _, _ = get_cup_dataset(dev_set_size=1.0) # retrain again on the full dataset
    nn = get_model()
    nn.train(
        X_full, 
        y_full,
        epochs=final_epochs,
        batch_size=BATCH_SIZE,
        verbose=False,
    )
    X_test_final = get_cup_final_test_set()
    save_final_prediction(nn.predict(X_test_final))
