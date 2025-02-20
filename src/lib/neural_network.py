import numpy as np
import matplotlib.pyplot as plt

from lib.activations import activation_functions, activation_derivatives
from lib.losses import loss_functions, loss_derivatives
from lib.regularization import compute_reg_gradient, compute_reg_loss

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, lambda_reg=0.001, reg_type="l2",
                 loss_function_name=None, activation_function_names=None,
                 task="classification", lr_decay_type="none", decay_rate=0.0,
                 weight_init="base", momentum_type="none", momentum_alpha=0.9,
                 seed=42):
        """
        Initialize the neural network with architecture and hyperparameters.
        Pass a seed to ensure parameter initialization is self-contained.
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

        if momentum_type not in {"none", "momentum", "nesterov momentum"}:
            raise ValueError("momentum_type must be 'none', 'momentum', or 'nesterov momentum'.")
        self.momentum_type = momentum_type
        self.momentum_alpha = momentum_alpha if momentum_type != "none" else 0.0

        if self.task == "regression":
            self.loss_function_name = loss_function_name or "mse"
        else:
            self.loss_function_name = loss_function_name or "binary_crossentropy"

        if activation_function_names is None:
            if self.task == "regression":
                activation_function_names = ["relu"] * (len(layers) - 1)
                activation_function_names[-1] = "linear"
            else:
                activation_function_names = ["relu"] * (len(layers) - 1)
                activation_function_names[-1] = "sigmoid"
        else:
            if len(activation_function_names) != len(layers) - 1:
                raise ValueError("activation_function_names must have length equal to len(layers)-1.")
        self.activation_function_names = activation_function_names

        # Initialize weights using the given seed.
        self._initialize_parameters(seed)
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

        self.train_loss_history = []
        self.val_loss_history = None
        self.train_accuracy_history = [] if self.task == "classification" else None
        self.val_accuracy_history = None

    def _initialize_parameters(self, seed):
        self.W = []
        self.b = []
        np.random.seed(seed)
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            if self.weight_init == "base":
                std = np.sqrt(1.0 / fan_in)
            elif self.weight_init == "glorot":
                std = np.sqrt(2.0 / (fan_in + fan_out))
            else:
                raise ValueError("Unsupported weight initialization strategy. Use 'base' or 'glorot'.")
            weight = np.random.randn(fan_in, fan_out) * std
            self.W.append(weight)
            self.b.append(np.zeros((1, fan_out)))

    def _apply_activation(self, x, func_name):
        if func_name not in activation_functions:
            raise ValueError(f"Unsupported activation: {func_name}")
        return activation_functions[func_name](x)

    def _apply_activation_derivative(self, z, a, func_name):
        if func_name not in activation_derivatives:
            raise ValueError(f"Unsupported activation derivative: {func_name}")
        return activation_derivatives[func_name](z, a)

    def _forward(self, X, weights=None, biases=None):
        if weights is None:
            weights = self.W
        if biases is None:
            biases = self.b

        A = [X]
        Z = []
        for i in range(len(weights) - 1):
            z_curr = np.dot(A[-1], weights[i]) + biases[i]
            Z.append(z_curr)
            a_curr = self._apply_activation(z_curr, self.activation_function_names[i])
            A.append(a_curr)
        z_out = np.dot(A[-1], weights[-1]) + biases[-1]
        Z.append(z_out)
        a_out = self._apply_activation(z_out, self.activation_function_names[-1])
        A.append(a_out)
        return Z, A

    def _compute_gradients(self, X, y, Z, A, weights=None):
        if weights is None:
            weights = self.W
        m = X.shape[0]
        dA = loss_derivatives[self.loss_function_name](y, A[-1])
        dZ = dA * self._apply_activation_derivative(Z[-1], A[-1], self.activation_function_names[-1])
        reg_term = compute_reg_gradient(weights[-1], self.lambda_reg, self.reg_type, m)
        dW = [np.dot(A[-2].T, dZ) / m + reg_term]
        db = [np.sum(dZ, axis=0, keepdims=True) / m]

        for i in range(len(weights) - 2, -1, -1):
            dA = np.dot(dZ, weights[i + 1].T)
            dZ = dA * self._apply_activation_derivative(Z[i], A[i + 1], self.activation_function_names[i])
            reg_term = compute_reg_gradient(weights[i], self.lambda_reg, self.reg_type, m)
            dW.insert(0, np.dot(A[i].T, dZ) / m + reg_term)
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)

        return dW, db

    def train(self, X, y, epochs=300, batch_size=32, verbose=True,
              early_stopping=False, validation_data=None, patience=10, min_delta=0.0):
        self.train_loss_history = []
        self.val_loss_history = [] if validation_data is not None else None
        if self.task == "classification":
            self.train_accuracy_history = []
            if validation_data is not None:
                self.val_accuracy_history = []

        n_samples = X.shape[0]
        best_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            if self.lr_decay_type == "exponential":
                self.learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)
            elif self.lr_decay_type == "linear":
                self.learning_rate = self.initial_learning_rate * max(0, 1 - self.decay_rate * epoch)

            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                if self.momentum_type == "nesterov momentum":
                    weights_lookahead = [self.W[j] - self.momentum_alpha * self.vW[j] for j in range(len(self.W))]
                    biases_lookahead = [self.b[j] - self.momentum_alpha * self.vb[j] for j in range(len(self.b))]
                    Z, A = self._forward(X_batch, weights=weights_lookahead, biases=biases_lookahead)
                    dW, db = self._compute_gradients(X_batch, y_batch, Z, A, weights=weights_lookahead)
                    for j in range(len(self.W)):
                        self.vW[j] = self.momentum_alpha * self.vW[j] + self.learning_rate * dW[j]
                        self.vb[j] = self.momentum_alpha * self.vb[j] + self.learning_rate * db[j]
                        self.W[j] -= self.vW[j]
                        self.b[j] -= self.vb[j]

                elif self.momentum_type == "momentum":
                    Z, A = self._forward(X_batch)
                    dW, db = self._compute_gradients(X_batch, y_batch, Z, A)
                    for j in range(len(self.W)):
                        self.vW[j] = self.momentum_alpha * self.vW[j] + self.learning_rate * dW[j]
                        self.vb[j] = self.momentum_alpha * self.vb[j] + self.learning_rate * db[j]
                        self.W[j] -= self.vW[j]
                        self.b[j] -= self.vb[j]

                else:
                    Z, A = self._forward(X_batch)
                    dW, db = self._compute_gradients(X_batch, y_batch, Z, A)
                    for j in range(len(self.W)):
                        self.W[j] -= self.learning_rate * dW[j]
                        self.b[j] -= self.learning_rate * db[j]

            # Compute training loss and (if applicable) validation loss.
            _, A_full = self._forward(X)
            train_loss = loss_functions[self.loss_function_name](y, A_full[-1])
            reg_loss = compute_reg_loss(self.W, self.lambda_reg, self.reg_type)
            total_train_loss = train_loss + reg_loss
            self.train_loss_history.append(total_train_loss)

            total_val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                _, A_val = self._forward(X_val)
                val_loss = loss_functions[self.loss_function_name](y_val, A_val[-1])
                reg_loss_val = compute_reg_loss(self.W, self.lambda_reg, self.reg_type)
                total_val_loss = val_loss + reg_loss_val
                self.val_loss_history.append(total_val_loss)

            if self.task == "classification":
                train_acc = self.evaluate(X, y)
                self.train_accuracy_history.append(train_acc)
                if validation_data is not None:
                    val_acc = self.evaluate(X_val, y_val)
                    self.val_accuracy_history.append(val_acc)

            if verbose:
                if total_val_loss is not None and self.task == "classification":
                    print(f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                          f"Validation Loss: {total_val_loss:.4f}, Training Acc: {train_acc:.4f}, "
                          f"Validation Acc: {val_acc:.4f}, Learning Rate: {self.learning_rate:.6f}")
                elif self.task == "classification":
                    print(f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                          f"Training Acc: {train_acc:.4f}, Learning Rate: {self.learning_rate:.6f}")
                elif total_val_loss is not None:
                    print(f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, "
                          f"Validation Loss: {total_val_loss:.4f}, Learning Rate: {self.learning_rate:.6f}")
                else:
                    print(f"Epoch {epoch:4d}, Training Loss: {total_train_loss:.4f}, Learning Rate: {self.learning_rate:.6f}")

            if early_stopping and validation_data is not None:
                if total_val_loss < best_loss - min_delta:
                    best_loss = total_val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.W]
                    best_biases = [b.copy() for b in self.b]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}. Restoring best model parameters.")
                        if best_weights is not None:
                            self.W = best_weights
                            self.b = best_biases
                        break

    def plot_loss_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label="Training Loss", color='blue')
        if self.val_loss_history is not None and len(self.val_loss_history) > 0:
            plt.plot(self.val_loss_history, label="Validation Loss", color='red', linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.show()

    def plot_accuracy_history(self):
        if self.task != "classification" or self.train_accuracy_history is None:
            print("Accuracy history is only available for classification tasks.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracy_history, label="Training Accuracy", color='blue')
        if self.val_accuracy_history is not None and len(self.val_accuracy_history) > 0:
            plt.plot(self.val_accuracy_history, label="Validation Accuracy", color='red', linestyle='dashed')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy History")
        plt.legend()
        plt.show()

    def predict(self, X):
        _, A = self._forward(X)
        output = A[-1]
        if self.task == "classification":
            if output.shape[1] == 1:
                return (output > 0.5).astype(int)
            else:
                return np.argmax(output, axis=1)
        else:
            return output

    def evaluate(self, X, y):
        predictions = self.predict(X)
        if self.task == "regression":
            return loss_functions["mee"](y, predictions)
        else:
            if y.ndim > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
            return np.mean(predictions == y_true)
