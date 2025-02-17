import numpy as np
import pandas as pd
from lib.data_loader import get_monks_dataset

# parametri per la rete neurale
hidden_layers = [10, 8]  # lista che specifica il numero di neuroni per ogni hidden layer
learning_rate = 0.05  # tasso di apprendimento
epochs = 1000  # numero di epoche di addestramento
batch_size = 32  # dimensione del batch per la discesa del gradiente
loss_function = "mse"  # funzione di perdita
activation_function = "relu"  # funzione di attivazione

# caricamento dataset
X_train, y_train, X_test, y_test = get_monks_dataset(3, one_hot_encode=True)
input_size = X_train.shape[1]  # numero di feature in ingresso
output_size = y_train.shape[1]  # numero di classi in output

# inizializzazione pesi e bias
np.random.seed(42)  # fissiamo il seed per la riproducibilità
layers = [input_size] + hidden_layers + [output_size]  # struttura della rete
W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]  # inizializzazione pesi
b = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]  # inizializzazione bias

# funzioni di attivazione
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # funzione sigmoide

def sigmoid_derivative(x):
    return x * (1 - x)  # derivata della sigmoide

def relu(x):
    return np.maximum(0, x)  # funzione ReLU

def relu_derivative(x):
    return (x > 0).astype(float)  # derivata della ReLU

def activation(x):
    return sigmoid(x) if activation_function == "sigmoid" else relu(x)  # scelta della funzione di attivazione

def activation_derivative(x):
    return sigmoid_derivative(x) if activation_function == "sigmoid" else relu_derivative(x)  # derivata della funzione di attivazione

# forward propagation
def forward_propagation(X):
    A = [X]  # lista per i valori attivati dei layer
    Z = []  # lista per i valori lineari dei layer
    for i in range(len(W) - 1):
        Z.append(np.dot(A[-1], W[i]) + b[i])  # calcolo del valore lineare
        A.append(activation(Z[-1]))  # applicazione della funzione di attivazione
    
    Z.append(np.dot(A[-1], W[-1]) + b[-1])  # calcolo per l'output layer
    A.append(sigmoid(Z[-1]))  # output layer usa sigmoide per classificazione
    return Z, A

# funzioni di perdita
def binary_crossentropy_loss(y_true, y_pred):
    epsilon = 1e-8  # valore per evitare log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clipping per evitare errori numerici
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # calcolo della loss entropia binaria

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)  # calcolo della loss MSE

def compute_loss(y_true, y_pred):
    return binary_crossentropy_loss(y_true, y_pred) if loss_function == "binary_crossentropy" else mse_loss(y_true, y_pred)  # scelta della funzione di perdita

# backward propagation
def backward_propagation(X, y, Z, A):
    global W, b
    m = X.shape[0]  # numero di esempi
    dZ = A[-1] - y  # errore dell'output layer
    
    dW = [np.dot(A[-2].T, dZ) / m]  # calcolo del gradiente per i pesi dell'output layer
    db = [np.sum(dZ, axis=0, keepdims=True) / m]  # calcolo del gradiente per i bias dell'output layer
    
    # propagazione all'indietro per i layer nascosti
    for i in range(len(W) - 2, -1, -1):
        dZ = np.dot(dZ, W[i+1].T) * activation_derivative(A[i+1])  # calcolo dell'errore nei layer nascosti
        dW.insert(0, np.dot(A[i].T, dZ) / m)  # aggiornamento gradiente pesi
        db.insert(0, np.sum(dZ, axis=0, keepdims=True) / m)  # aggiornamento gradiente bias
    
    # aggiornamento dei pesi e bias
    for i in range(len(W)):
        W[i] -= learning_rate * dW[i]
        b[i] -= learning_rate * db[i]

# training
loss_history = []  # lista per memorizzare l'andamento della loss
for epoch in range(epochs):
    permutation = np.random.permutation(X_train.shape[0])  # mescoliamo i dati
    X_train_shuffled = X_train[permutation]  # riordiniamo X_train
    y_train_shuffled = y_train[permutation]  # riordiniamo y_train
    
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]  # estrazione batch di input
        y_batch = y_train_shuffled[i:i+batch_size]  # estrazione batch di output
        
        Z, A = forward_propagation(X_batch)  # forward propagation sul batch
        backward_propagation(X_batch, y_batch, Z, A)  # backward propagation e aggiornamento pesi
    
    if epoch % 10 == 0:  # ogni 10 epoche calcoliamo la loss
        loss = compute_loss(y_train, forward_propagation(X_train)[1][-1])  # calcolo della loss sull'intero dataset
        loss_history.append(loss)  # salviamo la loss
        print(f"Epoch {epoch}, Loss: {loss:.4f}")  # stampiamo la loss

# test
_, A_test = forward_propagation(X_test)  # forward propagation sui dati di test
predictions = (A_test[-1] > 0.5).astype(int)  # conversione delle probabilità in classi (0 o 1)
accuracy = np.mean(predictions == y_test)  # calcolo dell'accuratezza
print(f"Test Accuracy: {accuracy:.4f}")  # stampiamo l'accuratezza finale

# plot della loss nel tempo
pd.Series(loss_history).plot()