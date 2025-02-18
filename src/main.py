import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import for sklearn dataset and preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================
# Activation functions and their derivatives
# ============================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z, a):
    # Note: z is not used here; kept for uniform signature.
    return a * (1 - a)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(z, a):
    return (z > 0).astype(float)

def linear(x):
    return x

def linear_derivative(z, a):
    return np.ones_like(a)

activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear
}

activation_derivatives = {
    "sigmoid": lambda z, a: sigmoid_derivative(z, a),
    "relu": lambda z, a: relu_derivative(z, a),
    "linear": lambda z, a: linear_derivative(z, a)
}

# ============================
# Loss functions and their derivatives
# ============================

def binary_crossentropy_loss(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

loss_functions = {
    "binary_crossentropy": binary_crossentropy_loss,
    "mse": mse_loss
}

loss_derivatives = {
    "binary_crossentropy": binary_crossentropy_derivative,
    "mse": mse_derivative
}

# ============================
# Regularization functions (modular)
# ============================

def compute_reg_gradient(W, lambda_reg, reg_type, m):
    if reg_type == "l2":
        return lambda_reg * W / m
    elif reg_type == "l1":
        return lambda_reg * np.sign(W) / m
    else:
        return 0

def compute_reg_loss(W_list, lambda_reg, reg_type):
    if reg_type == "l2":
        return (lambda_reg / 2) * sum(np.sum(W ** 2) for W in W_list)
    elif reg_type == "l1":
        return lambda_reg * sum(np.sum(np.abs(W)) for W in W_list)
    else:
        return 0

# ============================
# Neural Network Class with Learning Rate Decay Options
# ============================

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, lambda_reg=0.001, reg_type="l2",
                 loss_function_name=None,
                 activation_function_name="relu",
                 output_activation_function_name=None,
                 activation_function_names=None,
                 task="classification",
                 lr_decay_type="none",  # Options: "none", "exponential", "linear"
                 decay_rate=0.0):
        """
        :param layers: List containing the size of each layer (input, hidden, output)
        :param learning_rate: Initial learning rate
        :param lambda_reg: Regularization coefficient
        :param reg_type: Type of regularization ("l2", "l1", or other for none)
        :param loss_function_name: Name of the loss function (if None, set based on task)
        :param activation_function_name: Activation to use for hidden layers (if activation_function_names not provided)
        :param output_activation_function_name: Activation for the output layer (if None, set based on task)
        :param activation_function_names: List of activation function names for each layer (length = len(layers)-1)
        :param task: "classification" or "regression"
        :param lr_decay_type: Learning rate decay strategy ("none", "exponential", "linear")
        :param decay_rate: Decay rate used in the learning rate schedule
        """
        self.layers = layers
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.task = task
        self.lr_decay_type = lr_decay_type
        self.decay_rate = decay_rate
        
        # Set defaults based on task
        if self.task == "regression":
            self.loss_function_name = loss_function_name or "mse"
            output_activation_function_name = output_activation_function_name or "linear"
        else:
            self.loss_function_name = loss_function_name or "binary_crossentropy"
            output_activation_function_name = output_activation_function_name or "sigmoid"
        
        # If no list of activations is provided, use the same activation for all hidden layers and set the output activation
        if activation_function_names is None:
            self.activation_function_names = [activation_function_name] * (len(layers) - 1)
            self.activation_function_names[-1] = output_activation_function_name
        else:
            if len(activation_function_names) != len(layers) - 1:
                raise ValueError("activation_function_names must have length equal to len(layers)-1.")
            self.activation_function_names = activation_function_names
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        self.W = []
        self.b = []
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            # He initialization (good for ReLU)
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            self.W.append(weight)
            self.b.append(np.zeros((1, self.layers[i + 1])))
    
    def _apply_activation(self, x, func_name):
        if func_name not in activation_functions:
            raise ValueError(f"Unsupported activation: {func_name}")
        return activation_functions[func_name](x)
    
    def _apply_activation_derivative(self, z, a, func_name):
        if func_name not in activation_derivatives:
            raise ValueError(f"Unsupported activation derivative: {func_name}")
        return activation_derivatives[func_name](z, a)
    
    def _forward(self, X):
        A = [X]
        Z = []
        # Forward propagation through hidden layers
        for i in range(len(self.W) - 1):
            z_curr = np.dot(A[-1], self.W[i]) + self.b[i]
            Z.append(z_curr)
            a_curr = self._apply_activation(z_curr, self.activation_function_names[i])
            A.append(a_curr)
        # Forward propagation through output layer
        z_out = np.dot(A[-1], self.W[-1]) + self.b[-1]
        Z.append(z_out)
        a_out = self._apply_activation(z_out, self.activation_function_names[-1])
        A.append(a_out)
        return Z, A
    
    def _backward(self, X, y, Z, A):
        m = X.shape[0]
        if self.loss_function_name not in loss_derivatives:
            raise ValueError(f"Unsupported loss derivative: {self.loss_function_name}")
        # Compute derivative of loss with respect to output activation
        dA = loss_derivatives[self.loss_function_name](y, A[-1])
        # Compute derivative with respect to z at output layer
        dZ = dA * self._apply_activation_derivative(Z[-1], A[-1], self.activation_function_names[-1])
        reg_term = compute_reg_gradient(self.W[-1], self.lambda_reg, self.reg_type, m)
        dW = [np.dot(A[-2].T, dZ) / m + reg_term]
        db = [np.sum(dZ, axis=0, keepdims=True) / m]
        
        # Backpropagation through hidden layers
        for i in range(len(self.W) - 2, -1, -1):
            dA = np.dot(dZ, self.W[i + 1].T)
            dZ = dA * self._apply_activation_derivative(Z[i], A[i + 1], self.activation_function_names[i])
            reg_term = compute_reg_gradient(self.W[i], self.lambda_reg, self.reg_type, m)
            dW.insert(0, np.dot(A[i].T, dZ) / m + reg_term)
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)
        
        # Update parameters using current learning rate
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=300, batch_size=32, verbose=True):
        loss_history = []
        for epoch in range(epochs):
            # Update learning rate according to the decay schedule
            if self.lr_decay_type == "exponential":
                self.learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * epoch)
            elif self.lr_decay_type == "linear":
                # Ensure learning rate does not go negative.
                self.learning_rate = self.initial_learning_rate * max(0, 1 - self.decay_rate * epoch)
            # Otherwise ("none"), keep the initial learning rate.
            
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                Z, A = self._forward(X_batch)
                self._backward(X_batch, y_batch, Z, A)
            if epoch % max(1, int(epochs / 20)) == 0:
                _, A_full = self._forward(X)
                loss = loss_functions[self.loss_function_name](y, A_full[-1])
                reg_loss = compute_reg_loss(self.W, self.lambda_reg, self.reg_type)
                total_loss = loss + reg_loss
                loss_history.append(total_loss)
                if verbose:
                    print(f"Epoch {epoch:4d}, Loss: {total_loss:.4f}, Learning Rate: {self.learning_rate:.6f}")
        return loss_history
    
    def predict(self, X):
        _, A = self._forward(X)
        output = A[-1]
        if self.task == "classification":
            # For binary classification, threshold at 0.5
            if output.shape[1] == 1:
                return (output > 0.5).astype(int)
            else:
                return np.argmax(output, axis=1)
        else:  # regression
            return output
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        if self.task == "regression":
            return mse_loss(y, predictions)
        else:
            # If y is one-hot encoded, convert to class labels
            if y.ndim > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
            return np.mean(predictions == y_true)

# ============================
# Testing on a sklearn Classification Dataset
# ============================

# Load the breast cancer dataset (binary classification)
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)  # reshape y to be a column vector

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the network architecture
input_size = X_train.shape[1]
hidden_units = 10
output_size = 1  # binary classification
layers = [input_size, hidden_units, output_size]

# Define activation functions for hidden and output layers
activation_funcs = ["relu", "sigmoid"]

# Create the NeuralNetwork instance with learning rate decay.
# Change lr_decay_type to "exponential", "linear", or "none" as desired.
nn_clf = NeuralNetwork(
    layers=layers,
    learning_rate=0.1,
    lambda_reg=0.001,
    reg_type="l2",
    loss_function_name="mse",
    activation_function_names=activation_funcs,
    task="classification",
    lr_decay_type="exponential",  # Try "exponential", "linear", or "none"
    decay_rate=0.001              # Adjust decay rate as needed
)

print("Training the neural network on the breast cancer dataset...")
loss_history = nn_clf.train(X_train, y_train, epochs=1000, batch_size=32, verbose=True)

accuracy = nn_clf.evaluate(X_test, y_test)
print(f"\nNeural Network Classification Accuracy: {accuracy:.4f}")

# Plot the training loss history
pd.Series(loss_history).plot(title="Training Loss History")
plt.xlabel("Checkpoint (Epochs)")
plt.ylabel("Loss")
plt.show()
