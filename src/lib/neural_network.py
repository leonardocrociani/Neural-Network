"""
The 🧠 NEURAL NETWORK class.
"""

import numpy as np
import matplotlib.pyplot as plt
from lib.activations import activation_functions, activation_derivatives
from lib.error_functions import error_functions, error_functions_derivatives  
from lib.regularization import compute_reg_gradient, compute_reg_loss
np.random.seed(420)
import warnings
warnings.simplefilter("error")  # generate errors whenever a warning is issued. For debugging purposes and for correctness.

class NeuralNetwork:
    def __init__(
        self,
        layers,
        learning_rate=0.01,
        lambda_reg=0.001,
        reg_type="l2",
        error_function_name=None,
        activation_function_names=None,
        task="classification",
        lr_decay_type="none",
        decay_rate=0.0,
        weight_init="base",
        momentum_type="none",
        momentum_alpha=0.9,
    ):
        """
        Initialize the neural network with the given parameters. 
        Args:
            layers: list, the number of units in each layer.
            learning_rate: float, the learning rate of the network.
            lambda_reg: float, the regularization parameter.
            reg_type: str, the type of regularization to use.
            error_function_name: str, the name of the error function to use.
            activation_function_names: list, the names of the activation functions to use.
            task: str, the type of task to perform (classification or regression).
            lr_decay_type: str, the type of learning rate decay to use.
            decay_rate: float, the decay rate of the learning rate.
            weight_init: str, the type of weight initialization to use.
            momentum_type: str, the type of momentum to use.
            momentum_alpha: float, the alpha parameter of the momentum.
        """
        self.layers = layers
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.task = task
        self.lr_decay_type = lr_decay_type
        self.decay_rate = decay_rate
        self.weight_init = weight_init

        if momentum_type not in {"none", "momentum"}:
            raise ValueError("momentum_type must be 'none' or 'momentum'.")
        self.momentum_type = momentum_type
        self.momentum_alpha = momentum_alpha if momentum_type != "none" else 0.0

        if self.task == "regression":
            self.error_function_name = error_function_name or "mse"
        else:
            self.error_function_name = error_function_name or "binary_crossentropy"

        if activation_function_names is None:
            if self.task == "regression":
                activation_function_names = ["relu"] * (len(layers) - 1)
                activation_function_names[-1] = "linear"
            else:
                activation_function_names = ["relu"] * (len(layers) - 1)
                activation_function_names[-1] = "sigmoid"
        else:
            if len(activation_function_names) != len(layers) - 1:
                raise ValueError(
                    f"activation_function_names [{len(activation_function_names)}] must have length equal to len(layers)-1 [{len(layers)-1}]"
                )
        self.activation_function_names = activation_function_names

        self._initialize_parameters()

        # needed for momentum (they are the weights and biases updated at the previous step), they are 0 at the beginning.
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

        self.train_error_history = []
        self.val_error_history = None
        self.train_accuracy_history = [] if self.task == "classification" else None
        self.val_accuracy_history = None

    def _initialize_parameters(self):
        """
        Initialize the weights and biases of the network.
        """
        self.W = []
        self.b = []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            if self.weight_init == "base":  # initialize with lecun
                std = np.sqrt(1.0 / fan_in)
            elif self.weight_init == "glorot":
                std = np.sqrt(2.0 / (fan_in + fan_out))
            elif self.weight_init == "he":
                std = np.sqrt(2.0 / fan_in)
            else:
                raise ValueError(
                    "Unsupported weight initialization strategy. Use 'base' (for lecun), 'glorot' or 'he'."
                )
            weight = np.random.randn(fan_in, fan_out) * std
            self.W.append(weight)
            self.b.append(np.zeros((1, fan_out)))

    def summary(self):
        """
        Print a summary of the neural network.
        """
        print("\n# ==== Neural Network Summary: ===== #")
        print(f"\tTask: {self.task.title()}")
        print(f"\tError Function: {self.error_function_name.title()}")
        print(f"\tRegularization: {self.reg_type.title()} (lambda={self.lambda_reg})")
        print(f"\tLearning Rate: {self.initial_learning_rate}")
        print(f"\tLearning Rate Decay: {self.lr_decay_type.title()} (decay_rate={self.decay_rate})")
        print(f"\tMomentum: {self.momentum_type.title()} (alpha={self.momentum_alpha})")
        print(f"\tWeight Initialization: {self.weight_init.title()}")
        print(f"\tArchitecture: {self.layers}")
        print(f"\tActivation Functions: {self.activation_function_names}")
        print("# ================================== #\n")

    def _apply_activation(self, x, func_name):
        """
        Simple helper function to apply an activation function to a given input.
        Args:
            x: np.ndarray, the input to the activation function.
            func_name: str, the name of the activation function to apply.
        Returns:
            np.ndarray, the output of the activation function.
        """
        if func_name not in activation_functions:
            raise ValueError(f"Unsupported activation: {func_name}")
        return activation_functions[func_name](x)

    def _apply_activation_derivative(self, z, a, func_name):
        """
        Simple helper function to apply the derivative of an activation function to a given input.
        Args:
            z: np.ndarray, the input to the activation function.
            a: np.ndarray, the output of the activation function.
            func_name: str, the name of the activation function to apply.
        Returns:
            np.ndarray, the output of the derivative of the activation function.
        """
        if func_name not in activation_derivatives:
            raise ValueError(f"Unsupported activation derivative: {func_name}")
        return activation_derivatives[func_name](z, a)

    def _forward(self, X):
        """
        Perform the forward pass of the neural network.
        Args:
            X: np.ndarray, the input data.
        Returns:
            tuple, containing the weighted outputs (Z) and the activations (A) for each layer.
        """      
        A = [X]
        Z = [] 

        for i in range(len(self.W)):
            # input * weights + bias
            z_curr = np.dot(A[-1], self.W[i]) + self.b[i]
            Z.append(z_curr)

            # activation
            a_curr = self._apply_activation(z_curr, self.activation_function_names[i])
            A.append(a_curr)

        return Z, A

    def _compute_gradients(self, X, y, Z, A):
        """
        Compute the gradients of the weights and biases of the network.
        Args:
            X: np.ndarray, the input data.
            y: np.ndarray, the target data.
            Z: list, the weighted outputs of the network.
            A: list, the activations of the network.
        Returns:
            tuple, containing the gradients of the weights and biases.
        """

        # number of examples in the batch
        l = X.shape[0]

        # error at the output layer (δ_k)
        delta = error_functions_derivatives[self.error_function_name](
            y, A[-1]
        ) * self._apply_activation_derivative(
            Z[-1], A[-1], self.activation_function_names[-1]
        )

        reg_term = compute_reg_gradient(self.W[-1], self.lambda_reg, self.reg_type)

        # gradient of weights for the output layer
        dW = [(np.dot(A[-2].T, delta) / l) + reg_term]

        # for the bias, we can see it as a special case of weights with input 1
        # so the bias gradient is the sum of the deltas. Axis=0 indicates that we are summing for each example, as delta has shape (l, n_output)
        # keepdims is used to keep the shape of the initial bias, instead of collapsing it into a vector of shape (n_output,)
        db = [np.sum(delta, axis=0, keepdims=True) / l]

        # we start from the output layer and go backwards to the first hidedn layer
        for i in range(len(self.W) - 2, -1, -1):

            # delta for the ith hidden layer: δ_j = (Σ δ_k * w_kj) * f'(net_j)
            delta = np.dot(delta, self.W[i + 1].T) * self._apply_activation_derivative(
                Z[i], A[i + 1], self.activation_function_names[i]
            )

            reg_term = compute_reg_gradient(self.W[i], self.lambda_reg, self.reg_type)

            # updating the gradient of the weights and biases for the i-th hidden layer
            dW.insert(0, (np.dot(A[i].T, delta) / l) + reg_term)
            db.insert(0, np.sum(delta, axis=0, keepdims=True) / l)

        return dW, db

    def train(
        self,
        X,
        y,
        epochs=300,
        batch_size=32,
        verbose=True,
        early_stopping=False,
        validation_data=None,
        patience=10,
        min_delta=0.0,
    ):
        """
        Train the neural network on the given data.
        Args:
            X: np.ndarray, the input data.
            y: np.ndarray, the target data.
            epochs: int, the number of epochs to train the network.
            batch_size: int, the size of the mini-batches.
            verbose: bool, whether to print information during training.
            early_stopping: bool, whether to use early stopping.
            validation_data: tuple, containing the validation data (X_val, y_val).
            patience: int, the number of epochs to wait before early stopping.
            min_delta: float, the minimum change in the validation loss to consider for early stopping.
        Returns:
            int, the epoch at which the training was stopped (if early stopping was used).
        """
        self.train_error_history = []
        self.val_error_history = [] if validation_data is not None else None

        if self.task == "classification":
            self.train_accuracy_history = []
            if validation_data is not None:
                self.val_accuracy_history = []

        n_samples = X.shape[0]

        if batch_size == "full":
            batch_size = n_samples

        # varibles for early stopping.
        best_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_biases = None
        early_stopped_epoch = None

        for epoch in range(epochs):

            # learning rate update (if specified by the initial params)
            if self.lr_decay_type == "exponential":
                self.learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)
            elif self.lr_decay_type == "linear":
                self.learning_rate = self.initial_learning_rate * max(
                    self.initial_learning_rate * 0.01, # 1% of the initial learning rate
                    1 - self.decay_rate * epoch
                )

            # shuffling the samples
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            try:
                for i in range(0, n_samples, batch_size):
                    # for each batch:
                    # - take the samples
                    # - compute the gradients
                    # - update the weights
                    X_batch = X_shuffled[i : i + batch_size]
                    y_batch = y_shuffled[i : i + batch_size]

                    if self.momentum_type == "momentum":
                        # update with momentum => vW and vb are used to store/retrieve the previous updates.
                        Z, A = self._forward(X_batch)
                        dW, db = self._compute_gradients(X_batch, y_batch, Z, A)
                        for j in range(len(self.W)):
                            self.vW[j] = (self.momentum_alpha * self.vW[j]) - (self.learning_rate * dW[j])
                            self.vb[j] = (self.momentum_alpha * self.vb[j]) - (self.learning_rate * db[j])
                            self.W[j] += self.vW[j]
                            self.b[j] += self.vb[j]

                    else:
                        # classic update
                        Z, A = self._forward(X_batch)
                        dW, db = self._compute_gradients(X_batch, y_batch, Z, A)
                        for j in range(len(self.W)):
                            self.W[j] -= self.learning_rate * dW[j]
                            self.b[j] -= self.learning_rate * db[j]
            except Exception as e:
                print("Error in arch:")
                self.summary()
                raise e

            # at this point I can save the loss and the accuracy on the training set.
            _, A_full = self._forward(X)
            train_error = error_functions[self.error_function_name](y, A_full[-1])
            train_reg_penalty = compute_reg_loss(self.W, self.lambda_reg, self.reg_type) / X.shape[0]
            total_train_loss = train_error + train_reg_penalty
            self.train_error_history.append(train_error)

            total_val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                _, A_val = self._forward(X_val)
                val_error = error_functions[self.error_function_name](y_val, A_val[-1])
                val_reg_penalty = compute_reg_loss(self.W, self.lambda_reg, self.reg_type) / X_val.shape[0]
                total_val_loss = val_error + val_reg_penalty
                self.val_error_history.append(val_error)

            if self.task == "classification":
                train_acc = self.evaluate(X, y)
                self.train_accuracy_history.append(train_acc)
                if validation_data is not None:
                    val_acc = self.evaluate(X_val, y_val)
                    self.val_accuracy_history.append(val_acc)

            if verbose:
                if total_val_loss is not None and self.task == "classification":
                    print(
                        f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                        f"Validation Loss: {total_val_loss:.4f}, Training Acc: {train_acc:.4f}, "
                        f"Validation Acc: {val_acc:.4f}, Learning Rate: {self.learning_rate:.6f}"
                    )
                elif self.task == "classification":
                    print(
                        f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                        f"Training Acc: {train_acc:.4f}, Learning Rate: {self.learning_rate:.6f}"
                    )
                elif total_val_loss is not None:
                    print(
                        f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                        f"Validation Loss: {total_val_loss:.4f}, Learning Rate: {self.learning_rate:.6f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, Learning Rate: {self.learning_rate:.6f}"
                    )

            if early_stopping and validation_data is not None:
                # check if the new loss is better than the best loss by at least min_delta.
                if total_val_loss < best_loss - min_delta:
                    best_loss = total_val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.W]
                    best_biases = [b.copy() for b in self.b]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        early_stopped_epoch = epoch
                        if verbose:
                            print(
                                f"Early stopping triggered at epoch {epoch}. Restoring best model parameters."
                            )
                        if best_weights is not None:
                            self.W = best_weights
                            self.b = best_biases
                        break

        return early_stopped_epoch

    def plot_learning_curve(self, save_path=None, zoomed=False):
        """
        Plot the learning curve of the neural network.
        Args:
            save_path: str, the path where to save the plot.
            zoomed: bool, whether to zoom the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_error_history, label="Training Loss", color="red")
        if self.val_error_history is not None and len(self.val_error_history) > 0:
            plt.plot(
                self.val_error_history,
                label="Validation Loss",
                color="blue",
                linestyle="dashed",
            )
        plt.xlabel("Epochs")
        plt.ylabel(f"{self.error_function_name.upper()}")
        plt.title(f"{self.error_function_name.upper()} History")
        plt.legend()
        if zoomed:
            max_y = max(max(self.train_error_history), max(self.val_error_history))
            plt.ylim(0, max_y / 2)
        if save_path is not None:
            plt.savefig(save_path)
        # plt.show()

    def plot_accuracy_history(self, save_path=None):
        """
        Plot the accuracy history of the neural network.
        Args:
            save_path: str, the path where to save the plot.
        """
        if self.task != "classification" or self.train_accuracy_history is None:
            print("Accuracy history is only available for classification tasks.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracy_history, label="Training Accuracy", color="red")
        if self.val_accuracy_history is not None and len(self.val_accuracy_history) > 0:
            plt.plot(
                self.val_accuracy_history,
                label="Validation Accuracy",
                color="blue",
                linestyle="dashed",
            )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy History")
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        # plt.show()

    def predict(self, X, discretize=True):
        """
        Make predictions using the neural network.
        Args:
            X: np.ndarray, the input data.
            discretize: bool, whether to discretize the output.
        Returns:
            np.ndarray, the predictions.
        """
        _, A = self._forward(X)
        output = A[-1]

        if not discretize:
            return output

        if self.task == "classification":
            if output.shape[1] == 1:
                return (output > 0.5).astype(int)
            else:
                return np.argmax(output, axis=1)
        else:
            return output

    def evaluate(self, X, y, evaluation_function_name=None, discretize=True):
        """
        Evaluate the neural network on the given data.
        Args:
            X: np.ndarray, the input data.
            y: np.ndarray, the target data.
            evaluation_function_name: str, the evaluation function to use.
            discretize: bool, whether to discretize the output.
        Returns:
            float, the evaluation score.
        """
        predictions = self.predict(X, discretize)
        if evaluation_function_name:
            return error_functions[evaluation_function_name](y, predictions)
        if self.task == "regression":
            return error_functions["mse"](y, predictions)
        else:
            # return accuracy
            if y.ndim > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
            return np.mean(predictions == y_true)