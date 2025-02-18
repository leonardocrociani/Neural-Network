import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import per il dataset e la baseline lineare
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ============================
# Funzioni di attivazione e derivate
# ============================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z, a):
    # z non viene usato, ma lo manteniamo per avere la stessa firma
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
# Funzioni di loss e derivate
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
# Funzioni per la regolarizzazione (modulari)
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
# Classe della Rete Neurale
# ============================

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, lambda_reg=0.001, reg_type="l2",
                 loss_function_name=None,
                 activation_function_name="relu",
                 output_activation_function_name=None,
                 activation_function_names=None,
                 task="classification"):
        """
        :param layers: lista con la dimensione di ogni layer (input, hidden, output)
        :param learning_rate: tasso di apprendimento
        :param lambda_reg: coefficiente di regolarizzazione
        :param reg_type: tipo di regolarizzazione ("l2" o "l1")
        :param loss_function_name: nome della funzione di loss (se None, viene settata in base al task)
        :param activation_function_name: attivazione da usare per i layer nascosti (se non viene specificato activation_function_names)
        :param output_activation_function_name: attivazione per il layer di output (se None, viene settata in base al task)
        :param activation_function_names: lista di nomi di funzioni di attivazione per ogni layer (lunghezza = len(layers)-1)
        :param task: "classification" o "regression"
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        self.task = task
        
        # Impostiamo i default in base al task
        if self.task == "regression":
            self.loss_function_name = loss_function_name or "mse"
            output_activation_function_name = output_activation_function_name or "linear"
        else:
            self.loss_function_name = loss_function_name or "binary_crossentropy"
            output_activation_function_name = output_activation_function_name or "sigmoid"
        
        # Se non è specificata una lista di attivazioni per ogni layer, usiamo la stessa per tutti i layer nascosti
        # e quella di output per l'ultimo layer
        if activation_function_names is None:
            # Creo una lista di lunghezza len(layers)-1
            self.activation_function_names = [activation_function_name] * (len(layers) - 1)
            self.activation_function_names[-1] = output_activation_function_name
        else:
            if len(activation_function_names) != len(layers) - 1:
                raise ValueError("La lista activation_function_names deve avere una lunghezza pari a len(layers)-1.")
            self.activation_function_names = activation_function_names
        
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        self.W = []
        self.b = []
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            # Inizializzazione di He (ottimale per ReLU)
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            self.W.append(weight)
            self.b.append(np.zeros((1, self.layers[i + 1])))
    
    def _apply_activation(self, x, func_name):
        if func_name not in activation_functions:
            raise ValueError(f"Attivazione non supportata: {func_name}")
        return activation_functions[func_name](x)
    
    def _apply_activation_derivative(self, z, a, func_name):
        if func_name not in activation_derivatives:
            raise ValueError(f"Derivata dell'attivazione non supportata: {func_name}")
        return activation_derivatives[func_name](z, a)
    
    def _forward(self, X):
        A = [X]
        Z = []
        # Propagazione attraverso i layer nascosti (tutti tranne l'output)
        for i in range(len(self.W) - 1):
            z_curr = np.dot(A[-1], self.W[i]) + self.b[i]
            Z.append(z_curr)
            a_curr = self._apply_activation(z_curr, self.activation_function_names[i])
            A.append(a_curr)
        # Propagazione nel layer di output
        z_out = np.dot(A[-1], self.W[-1]) + self.b[-1]
        Z.append(z_out)
        a_out = self._apply_activation(z_out, self.activation_function_names[-1])
        A.append(a_out)
        return Z, A
    
    def _backward(self, X, y, Z, A):
        m = X.shape[0]
        if self.loss_function_name not in loss_derivatives:
            raise ValueError(f"Derivata della loss non supportata: {self.loss_function_name}")
        # Calcola dL/dy_pred per l'output
        dA = loss_derivatives[self.loss_function_name](y, A[-1])
        # Calcola dL/dz per il layer di output
        dZ = dA * self._apply_activation_derivative(Z[-1], A[-1], self.activation_function_names[-1])
        reg_term = compute_reg_gradient(self.W[-1], self.lambda_reg, self.reg_type, m)
        dW = [np.dot(A[-2].T, dZ) / m + reg_term]
        db = [np.sum(dZ, axis=0, keepdims=True) / m]
        
        # Backpropagation nei layer nascosti
        for i in range(len(self.W) - 2, -1, -1):
            dA = np.dot(dZ, self.W[i + 1].T)
            dZ = dA * self._apply_activation_derivative(Z[i], A[i + 1], self.activation_function_names[i])
            reg_term = compute_reg_gradient(self.W[i], self.lambda_reg, self.reg_type, m)
            dW.insert(0, np.dot(A[i].T, dZ) / m + reg_term)
            db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)
        
        # Aggiorna i parametri
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=300, batch_size=32, verbose=True):
        loss_history = []
        for epoch in range(epochs):
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
                    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        return loss_history
    
    def predict(self, X):
        _, A = self._forward(X)
        output = A[-1]
        if self.task == "classification":
            if output.shape[1] == 1:
                return (output > 0.5).astype(int)
            else:
                return np.argmax(output, axis=1)
        else:  # regressione
            return output
    
    def evaluate(self, X, y):
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
# Esperimento di regressione con il dataset Diabetes
# ============================

# Caricamento e scaling del dataset
data = load_diabetes()
X = data.data
y = data.target.reshape(-1, 1)  # rendiamo y bidimensionale

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Suddividiamo in train e test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Definiamo l'architettura della rete per regressione
input_size = X_train.shape[1]    # 10 feature
layers = [input_size, 10, 10, 1]   # 2 hidden layer con 10 neuroni ciascuno, 1 output

# Esempio: specifica di funzioni di attivazione diverse per ogni layer
# In questo caso, per i due layer nascosti usiamo "relu" e per l'output "linear"
activation_funcs = ["relu", "relu", "linear"]

# Istanziamo la rete neurale per regressione
nn_reg = NeuralNetwork(
    layers=layers,
    learning_rate=0.001,
    lambda_reg=0.001,
    reg_type="l2",
    loss_function_name="mse",  # per regressione usiamo MSE
    activation_function_names=activation_funcs,
    task="regression"
)

print("Training della rete neurale per regressione...")
loss_history = nn_reg.train(X_train, y_train, epochs=300, batch_size=32, verbose=True)
nn_mse = nn_reg.evaluate(X_test, y_test)
print(f"\nNeural Network Test MSE (sul target scalato): {nn_mse:.4f}")

# Baseline: regressione lineare di sklearn
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_mse = np.mean((y_test - y_pred_lr) ** 2)
print(f"Baseline Linear Regression Test MSE (sul target scalato): {lr_mse:.4f}")

pd.Series(loss_history).plot()
plt.xlabel("Epoch (intervalli)")
plt.ylabel("Loss")
plt.title("Loss History")
plt.show()
