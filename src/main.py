import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid  # Using scikit-learn, but just  for cross-validation and grid search SETUP (split or combination generation).
from joblib import Parallel, delayed  # For running grid search evaluations in parallel.

from lib.data_loader import get_monks_dataset  # Custom function to load one of the Monks datasets.

# ============================
# Activation functions and their derivatives
# ============================
# These functions define common activation functions used in neural networks,
# along with their derivatives needed for backpropagation.

def sigmoid(x):
    # Compute the sigmoid activation: 1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z, a):
    # Compute derivative of sigmoid using the already computed activation value.
    # Note: z (the pre-activation) is not used here; it is kept to maintain a uniform signature.
    return a * (1 - a)

def relu(x):
    # Compute the Rectified Linear Unit (ReLU): max(0, x)
    return np.maximum(0, x)

def relu_derivative(z, a):
    # Derivative of ReLU: 1 if z > 0, else 0.
    # Using z (the pre-activation) here to decide where the gradient flows.
    return (z > 0).astype(float)

def linear(x):
    # Linear activation (identity function): useful for regression output layers.
    return x

def linear_derivative(z, a):
    # Derivative of a linear function is constant 1.
    return np.ones_like(a)

# Dictionaries to map activation function names to their implementations.
activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear
}

# Similarly, mapping names to derivative functions.
activation_derivatives = {
    "sigmoid": lambda z, a: sigmoid_derivative(z, a),
    "relu": lambda z, a: relu_derivative(z, a),
    "linear": lambda z, a: linear_derivative(z, a)
}

# ============================
# Loss functions and their derivatives
# ============================
# Here we define loss functions for different tasks (binary classification and regression)
# along with their derivatives for backpropagation.

def binary_crossentropy_loss(y_true, y_pred):
    """
    Compute binary crossentropy loss for binary classification.
    Clipping is used to avoid taking the log of zero.
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    """
    Derivative of the binary crossentropy loss with respect to predictions.
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def mse_loss(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) loss, typically used for regression tasks.
    """
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    """
    Derivative of the Mean Squared Error (MSE) loss.
    """
    return 2 * (y_pred - y_true)

def mee_loss(y_true, y_pred):
    """
    Compute Mean Euclidean Error (MEE) loss:
    For each sample, calculates the Euclidean distance between y_true and y_pred.
    """
    diff = y_true - y_pred  # Difference between true and predicted values.
    # Compute Euclidean distance for each sample.
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(dist)

def mee_derivative(y_true, y_pred):
    """
    Compute derivative of the Mean Euclidean Error (MEE) loss.
    For each sample, the derivative is (1/N) * ((y_pred - y_true) / ||y_pred - y_true||).
    An epsilon is used to avoid division by zero.
    """
    diff = y_pred - y_true
    dist = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True))
    epsilon = 1e-8  # Small constant to avoid division by zero.
    dist_safe = np.where(dist == 0, epsilon, dist)
    N = y_true.shape[0]
    derivative = diff / dist_safe / N
    return derivative

# Dictionaries to map loss function names to their implementations.
loss_functions = {
    "binary_crossentropy": binary_crossentropy_loss,
    "mse": mse_loss,
    "mee": mee_loss,  
}

loss_derivatives = {
    "binary_crossentropy": binary_crossentropy_derivative,
    "mse": mse_derivative,
    "mee": mee_derivative, 
}

# ============================
# Regularization functions (modular)
# ============================
# Functions to compute regularization loss and gradients (L1 and L2) to avoid overfitting.

def compute_reg_gradient(W, lambda_reg, reg_type, m):
    # Compute regularization gradient for a weight matrix W.
    if reg_type == "l2":
        return lambda_reg * W / m
    elif reg_type == "l1":
        return lambda_reg * np.sign(W) / m
    else:
        return 0

def compute_reg_loss(W_list, lambda_reg, reg_type):
    # Compute the total regularization loss over a list of weight matrices.
    if reg_type == "l2":
        return (lambda_reg / 2) * sum(np.sum(W ** 2) for W in W_list)
    elif reg_type == "l1":
        return lambda_reg * sum(np.sum(np.abs(W)) for W in W_list)
    else:
        return 0

# ============================
# Neural Network Class with Learning Rate Decay, Momentum, Custom Weight Initialization, and Early Stopping
# ============================
# This class implements a feed-forward neural network with several advanced features.
# I designed it to be flexible and modular so I can experiment with different training strategies.

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, lambda_reg=0.001, reg_type="l2",
                 loss_function_name=None,
                 activation_function_name="relu",
                 output_activation_function_name=None,
                 activation_function_names=None,
                 task="classification",
                 lr_decay_type="none",  # Options: "none", "exponential", "linear"
                 decay_rate=0.0,
                 weight_init="base",  # "base" (fan-in scaling) or "glorot"
                 momentum_type="none",  # Options: "none", "momentum", "nesterov momentum"
                 momentum_alpha=0.9):
        """
        Initialize the neural network with architecture and training hyperparameters.
        
        Parameters:
            layers: List with the number of neurons per layer (input, hidden, output).
            learning_rate: Starting learning rate.
            lambda_reg: Regularization coefficient.
            reg_type: Regularization type ("l2", "l1", or none).
            loss_function_name: Loss function to use (set based on task if None).
            activation_function_name: Default activation for hidden layers.
            output_activation_function_name: Activation for the output layer (set based on task if None).
            activation_function_names: List of activations per layer (overrides defaults if provided).
            task: "classification" or "regression".
            lr_decay_type: Learning rate decay strategy.
            decay_rate: Decay rate factor.
            weight_init: Weight initialization strategy.
            momentum_type: Momentum strategy.
            momentum_alpha: Momentum coefficient.
        """
        self.layers = layers
        self.initial_learning_rate = learning_rate  # Save the initial learning rate for decay computations.
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.task = task
        self.lr_decay_type = lr_decay_type
        self.decay_rate = decay_rate
        self.weight_init = weight_init
        
        # Set momentum parameters; validate the input.
        if momentum_type not in {"none", "momentum", "nesterov momentum"}:
            raise ValueError("momentum_type must be 'none', 'momentum', or 'nesterov momentum'.")
        self.momentum_type = momentum_type
        self.momentum_alpha = momentum_alpha if momentum_type != "none" else 0.0
        
        # Set default loss and output activation based on the task.
        if self.task == "regression":
            self.loss_function_name = loss_function_name or "mse"
            output_activation_function_name = output_activation_function_name or "linear"
        else:
            # For classification tasks.
            self.loss_function_name = loss_function_name or "binary_crossentropy"
            output_activation_function_name = output_activation_function_name or "sigmoid"
        
        # Set activation functions for each layer.
        if activation_function_names is None:
            # Use a default activation for all hidden layers and a specific one for the output layer.
            self.activation_function_names = [activation_function_name] * (len(layers) - 1)
            self.activation_function_names[-1] = output_activation_function_name
        else:
            if len(activation_function_names) != len(layers) - 1:
                raise ValueError("activation_function_names must have length equal to len(layers)-1.")
            self.activation_function_names = activation_function_names
        
        # Initialize weights and biases for the network.
        self._initialize_parameters()
        # Initialize momentum accumulators (even if not used, for consistency).
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        
        # Loss history lists (these will be reinitialized in train()).
        self.train_loss_history = []
        self.val_loss_history = None

    def _initialize_parameters(self):
        """
        Initialize the weights and biases for each layer using either base (fan-in) or Glorot initialization.
        """
        self.W = []
        self.b = []
        np.random.seed(42)  # For reproducibility.
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            if self.weight_init == "base":
                std = np.sqrt(1.0 / fan_in)
            elif self.weight_init == "glorot":
                std = np.sqrt(2.0 / (fan_in + fan_out))
            else:
                raise ValueError("Unsupported weight initialization strategy. Use 'base' or 'glorot'.")
            weight = np.random.randn(fan_in, fan_out) * std  # Initialize weights with Gaussian noise.
            self.W.append(weight)
            self.b.append(np.zeros((1, fan_out)))  # Biases are initialized to zero.
    
    def _apply_activation(self, x, func_name):
        """
        Helper to apply an activation function based on its name.
        """
        if func_name not in activation_functions:
            raise ValueError(f"Unsupported activation: {func_name}")
        return activation_functions[func_name](x)
    
    def _apply_activation_derivative(self, z, a, func_name):
        """
        Helper to apply the derivative of an activation function based on its name.
        """
        if func_name not in activation_derivatives:
            raise ValueError(f"Unsupported activation derivative: {func_name}")
        return activation_derivatives[func_name](z, a)
    
    def _forward(self, X, weights=None, biases=None):
        """
        Forward propagation through the network.
        If custom weights and biases are provided (e.g., for Nesterov momentum lookahead), use them.
        Returns:
            Z: List of pre-activation values for each layer.
            A: List of activations for each layer (including input and output).
        """
        if weights is None:
            weights = self.W
        if biases is None:
            biases = self.b
            
        A = [X]  # Start with the input layer.
        Z = []
        # Propagate through hidden layers.
        for i in range(len(weights) - 1):
            z_curr = np.dot(A[-1], weights[i]) + biases[i]
            Z.append(z_curr)
            a_curr = self._apply_activation(z_curr, self.activation_function_names[i])
            A.append(a_curr)
        # Process the output layer separately.
        z_out = np.dot(A[-1], weights[-1]) + biases[-1]
        Z.append(z_out)
        a_out = self._apply_activation(z_out, self.activation_function_names[-1])
        A.append(a_out)
        return Z, A
    
    def _compute_gradients(self, X, y, Z, A, weights=None):
        """
        Compute gradients for weights and biases via backpropagation.
        
        Parameters:
            X: Input batch.
            y: True labels/targets.
            Z: Pre-activation values collected during forward pass.
            A: Activations collected during forward pass.
            weights: Optional custom weights (e.g., for lookahead in Nesterov momentum).
        
        Returns:
            dW: List of gradients for weight matrices.
            db: List of gradients for bias vectors.
        """
        if weights is None:
            weights = self.W
        m = X.shape[0]
        # Compute derivative of loss w.r.t. output activation.
        dA = loss_derivatives[self.loss_function_name](y, A[-1])
        # For the output layer.
        dZ = dA * self._apply_activation_derivative(Z[-1], A[-1], self.activation_function_names[-1])
        # Regularization term for the output layer.
        reg_term = compute_reg_gradient(weights[-1], self.lambda_reg, self.reg_type, m)
        dW = [np.dot(A[-2].T, dZ) / m + reg_term]
        db = [np.sum(dZ, axis=0, keepdims=True) / m]
        
        # Backpropagate through hidden layers.
        for i in range(len(weights) - 2, -1, -1):
            dA = np.dot(dZ, weights[i + 1].T)
            dZ = dA * self._apply_activation_derivative(Z[i], A[i + 1], self.activation_function_names[i])
            reg_term = compute_reg_gradient(weights[i], self.lambda_reg, self.reg_type, m)
            dW.insert(0, np.dot(A[i].T, dZ) / m + reg_term)
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)
            
        return dW, db
    
    def train(self, X, y, epochs=300, batch_size=32, verbose=True,
              early_stopping=False, validation_data=None, patience=10, min_delta=0.0):
        """
        Train the neural network using mini-batch gradient descent.
        
        Features include:
         - Learning rate decay (exponential or linear).
         - Momentum (classic or Nesterov).
         - Early stopping based on validation loss.
        
        The training and validation loss histories are stored in:
            self.train_loss_history and self.val_loss_history.
        """
        # Reinitialize loss histories at the start of training.
        self.train_loss_history = []
        if validation_data is not None:
            self.val_loss_history = []
        else:
            self.val_loss_history = None

        n_samples = X.shape[0]
        best_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            # Update learning rate based on the selected decay strategy.
            if self.lr_decay_type == "exponential":
                self.learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)
            elif self.lr_decay_type == "linear":
                self.learning_rate = self.initial_learning_rate * max(0, 1 - self.decay_rate * epoch)
            # Otherwise, if "none", the learning rate remains constant.
            
            # Shuffle training data to ensure randomness in mini-batches.
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            # Process the data in mini-batches.
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                if self.momentum_type == "nesterov momentum":
                    # Lookahead step: adjust current parameters with momentum before computing gradients.
                    weights_lookahead = [self.W[j] - self.momentum_alpha * self.vW[j] for j in range(len(self.W))]
                    biases_lookahead = [self.b[j] - self.momentum_alpha * self.vb[j] for j in range(len(self.b))]
                    Z, A = self._forward(X_batch, weights=weights_lookahead, biases=biases_lookahead)
                    dW, db = self._compute_gradients(X_batch, y_batch, Z, A, weights=weights_lookahead)
                    # Update momentum accumulators and then update parameters.
                    for j in range(len(self.W)):
                        self.vW[j] = self.momentum_alpha * self.vW[j] + self.learning_rate * dW[j]
                        self.vb[j] = self.momentum_alpha * self.vb[j] + self.learning_rate * db[j]
                        self.W[j] -= self.vW[j]
                        self.b[j] -= self.vb[j]
                        
                elif self.momentum_type == "momentum":
                    # Standard momentum update.
                    Z, A = self._forward(X_batch)
                    dW, db = self._compute_gradients(X_batch, y_batch, Z, A)
                    for j in range(len(self.W)):
                        self.vW[j] = self.momentum_alpha * self.vW[j] + self.learning_rate * dW[j]
                        self.vb[j] = self.momentum_alpha * self.vb[j] + self.learning_rate * db[j]
                        self.W[j] -= self.vW[j]
                        self.b[j] -= self.vb[j]
                        
                else:  # No momentum.
                    Z, A = self._forward(X_batch)
                    dW, db = self._compute_gradients(X_batch, y_batch, Z, A)
                    for j in range(len(self.W)):
                        self.W[j] -= self.learning_rate * dW[j]
                        self.b[j] -= self.learning_rate * db[j]
            
            # Compute training loss over the full training set.
            _, A_full = self._forward(X)
            train_loss = loss_functions[self.loss_function_name](y, A_full[-1])
            reg_loss = compute_reg_loss(self.W, self.lambda_reg, self.reg_type)
            total_train_loss = train_loss + reg_loss
            self.train_loss_history.append(total_train_loss)
            
            # If validation data is provided, compute the validation loss.
            if validation_data is not None:
                X_val, y_val = validation_data
                _, A_val = self._forward(X_val)
                val_loss = loss_functions[self.loss_function_name](y_val, A_val[-1])
                reg_loss_val = compute_reg_loss(self.W, self.lambda_reg, self.reg_type)
                total_val_loss = val_loss + reg_loss_val
                self.val_loss_history.append(total_val_loss)
            else:
                total_val_loss = None

            # Print progress information if verbosity is enabled.
            if verbose:
                if total_val_loss is not None:
                    print(f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                          f"Validation Loss: {total_val_loss:.4f}, Learning Rate: {self.learning_rate:.6f}")
                else:
                    print(f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                          f"Learning Rate: {self.learning_rate:.6f}")
            
            # Check for early stopping if enabled and using validation data.
            if early_stopping and (validation_data is not None):
                if total_val_loss < best_loss - min_delta:
                    best_loss = total_val_loss
                    patience_counter = 0
                    # Save the best parameters so far.
                    best_weights = [w.copy() for w in self.W]
                    best_biases = [b.copy() for b in self.b]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}. Restoring best model parameters.")
                        if best_weights is not None:
                            self.W = best_weights
                            self.b = best_biases
                        break

    def plot_loss_history(self):
        """
        Plot the loss history for both training and (if available) validation.
        Useful for visualizing convergence.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label="Training Loss")
        if self.val_loss_history is not None and len(self.val_loss_history) > 0:
            plt.plot(self.val_loss_history, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.show()
    
    def predict(self, X):
        """
        Generate predictions for given input X.
        For classification tasks, threshold or argmax is used as needed.
        """
        _, A = self._forward(X)
        output = A[-1]
        if self.task == "classification":
            if output.shape[1] == 1:
                return (output > 0.5).astype(int)  # Binary classification threshold.
            else:
                return np.argmax(output, axis=1)  # Multi-class classification.
        else:
            return output
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance.
        For regression, returns MSE.
        For classification, returns accuracy.
        """
        predictions = self.predict(X)
        if self.task == "regression":
            return mse_loss(y, predictions)
        else:
            if y.ndim > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
            return np.mean(predictions == y_true)

# ============================
# K-fold Cross-Validation Function
# ============================
# This function implements k-fold cross-validation.
# It splits the data, trains a new model for each fold, and aggregates the evaluation metrics.

def k_fold_cross_validation(model_builder, X, y, k=5, epochs=1000, batch_size=32,
                            verbose=True, early_stopping=False, patience=10, min_delta=1e-4):
    """
    Perform k-fold cross validation for a neural network.
    
    Parameters:
        model_builder: A callable that returns a new instance of NeuralNetwork.
        X: Input features.
        y: Targets.
        k: Number of folds.
        epochs: Number of training epochs per fold.
        batch_size: Batch size.
        verbose: Verbosity flag.
        early_stopping: Whether to use early stopping.
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum change in loss to be considered an improvement.
    
    Returns:
        A dictionary with per-fold metrics and the overall average metric.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"\nFold {fold}/{k}")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Build a new model instance for this fold.
        nn_model = model_builder()
        
        # Train the model using the current fold's training and validation data.
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
        
        # Evaluate performance on the validation fold.
        metric = nn_model.evaluate(X_val_fold, y_val_fold)
        print(f"Fold {fold} Evaluation Metric: {metric:.4f}")
        fold_metrics.append(metric)
        fold += 1
    
    avg_metric = np.mean(fold_metrics)
    print(f"\nAverage Evaluation Metric over {k} folds: {avg_metric:.4f}")
    return {"fold_metrics": fold_metrics, "average_metric": avg_metric}

# ============================
# Grid Search Function with Parallelism
# ============================
# This function performs a grid search over a set of hyperparameters.
# It uses k-fold cross-validation to evaluate each hyperparameter combination
# and runs evaluations in parallel for speed.

def grid_search(model_builder, param_grid, X, y, k=5, epochs=1000, batch_size=32,
                early_stopping=False, patience=10, min_delta=1e-4, n_jobs=-1,
                maximize=True, verbose=True):
    """
    Perform grid search over the given parameter grid using k-fold cross validation.
    
    Parameters:
        model_builder: A callable that accepts hyperparameters as keyword arguments and returns a NeuralNetwork.
        param_grid: Dictionary of hyperparameters to try.
        X: Input features.
        y: Targets.
        k: Number of folds for cross validation.
        epochs: Number of epochs per fold.
        batch_size: Batch size.
        early_stopping: Whether to use early stopping.
        patience: Patience for early stopping.
        min_delta: Minimum change in loss for early stopping.
        n_jobs: Number of parallel jobs (-1 uses all available processors).
        maximize: If True, a higher evaluation metric is better.
        verbose: Verbosity flag.
    
    Returns:
        A tuple (best_result, all_results) where best_result contains the best hyperparameters and metric.
    """
    grid = list(ParameterGrid(param_grid))
    results = []

    def evaluate_params(params):
        # Create a new model instance with the current hyperparameters.
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

    # Run evaluations in parallel.
    evaluated_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params) for params in grid
    )

    for params, metric in evaluated_results:
        results.append({"params": params, "average_metric": metric})

    # Choose the best hyperparameters based on whether a higher metric is better.
    if maximize:
        best_result = max(results, key=lambda x: x["average_metric"])
    else:
        best_result = min(results, key=lambda x: x["average_metric"])

    if verbose:
        print("\nGrid Search Results:")
        for res in results:
            print(f"Params: {res['params']}, Average Metric: {res['average_metric']:.4f}")
        print(f"\nBest Params: {best_result['params']}, Best Average Metric: {best_result['average_metric']:.4f}")

    return best_result, results

# ============================
# Testing on a Monk's Dataset
# ============================
# Load the dataset, configure the network architecture, and perform training,
# evaluation, cross-validation, and grid search.

# Load Monk's dataset (with one-hot encoding for targets).
X_train, y_train, X_test, y_test = get_monks_dataset(1, one_hot_encode=True)

# Define the network architecture.
input_size = X_train.shape[1]
hidden_units = 10
output_size = 1  # Binary classification problem.
layers = [input_size, hidden_units, output_size]

# Specify activation functions for the hidden and output layers.
activation_funcs = ["relu", "sigmoid"]

# Build a neural network classifier instance with the desired configuration.
nn_clf = NeuralNetwork(
    layers=layers,
    learning_rate=0.2,
    lambda_reg=0.001,
    reg_type="l2",
    loss_function_name="mse",       # Even for classification, trying MSE here.
    activation_function_names=activation_funcs,
    task="classification",
    lr_decay_type="linear",    # Options: "exponential", "linear", or "none"
    decay_rate=0.001,
    weight_init="base",        # Options: "base" or "glorot"
)

# Train the network.
# Early stopping is used here with the test set as validation data.
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

# Evaluate the network's classification accuracy.
accuracy = nn_clf.evaluate(X_test, y_test)
print(f"\nNeural Network Classification Accuracy: {accuracy:.4f}")

# Plot the training and validation loss history.
nn_clf.plot_loss_history()

# ============================
# K-fold Cross-Validation Example
# ============================
def build_nn_model():
    """
    Build and return a new instance of NeuralNetwork with the current configuration.
    This function is used to generate a fresh model for each cross-validation fold.
    """
    return NeuralNetwork(
        layers=layers,
        learning_rate=0.2,
        lambda_reg=0.001,
        reg_type="l2",
        loss_function_name="mse",       
        activation_function_names=activation_funcs,
        task="classification",
        lr_decay_type="linear",    # Options: "exponential", "linear", or "none"
        decay_rate=0.001,
        weight_init="base",        # Options: "base" or "glorot"
    )

# Perform 5-fold cross validation on the training set.
cv_results = k_fold_cross_validation(
    model_builder=build_nn_model,
    X=X_train,
    y=y_train,
    k=5,
    epochs=1000,
    batch_size=32,
    verbose=False,  # Set to True if you want detailed logging for each fold.
    early_stopping=True,
    patience=10,
    min_delta=1e-4
)

print("\nK-fold Cross-Validation Results:")
print(cv_results)

# ============================
# Grid Search Example
# ============================
# Define a model builder that accepts hyperparameters as keyword arguments.
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

# Define the hyperparameter grid to search over.
param_grid = {
    "learning_rate": [0.1, 0.2],
    "lambda_reg": [0.001, 0.01],
    "lr_decay_type": ["linear", "none"],
    "decay_rate": [0.001, 0.0],
    "weight_init": ["base", "glorot"],
}

# Perform grid search with 5-fold cross validation.
# Note: You can reduce epochs for grid search to speed up computation.
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