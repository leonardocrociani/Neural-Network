# Neural Network from Scratch

An implementation from scratch of a feed-forward neural network, featuring support for:

- **Weight initialization**: LeCun (“base”), Glorot, He  
- **Activation functions**: ReLU, Sigmoid, Linear  
- **Loss functions**: MSE (for regression), Binary Cross-Entropy (for classification)  
- **Regularization**: L1, L2  
- **Optimization options**: Momentum, Learning rate decay (none, exponential, linear)  
- **Training methods**: Batch and mini-batch gradient descent, early stopping  
- **Visualization**: Learning curve and accuracy history plots (for classification tasks)

---

The network was tested on the *Monks* and *Cup* datasets (from the Machine Learning course @ UniPi).  
A detailed report explaining the implementation and the results can be found in `doc/report.pdf`.

---

## Project Structure

```bash
Neural-Network/ 
├─ README.md 
├─ src/ 
│ ├─ lib/ 
│ │ ├─ activations.py
│ │ ├─ error_functions.py 
│ │ ├─ regularization.py 
│ │ └─ neural_network.py # ⟵ classe NeuralNetwork 
│ ├─ grid_search_monks.py # ricerca iperparametri Monks (classification) 
│ ├─ grid_search_cup.py # ricerca iperparametri Cup (regression) 
│ ├─ model_assessment_monks.py # valutazione modello Monks con migliori iperparametri 
│ └─ model_assessment_cup.py # valutazione modello Cup con migliori iperparametri 
├─ results/ # (plot, log, json dei risultati)
│ └─ hyperparams_search/ 
│   ├─ monks_1/… monks_2/… monks_3/… 
│   └─ cup/coarse/… cup/fine/… 
└─ doc/
    ├─ backprop.pdf
    └─ report.pdf
```

---

## Requirements

- Python ≥ 3.7  
- requirements.txt

> It's recommended to use a virtual environment.