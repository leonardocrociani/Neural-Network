# Doc...?

[Note: il codice va fatto partire dalla folder src/, per evitare che i dataset vengano riscaricati]

La rete è stata implementata per intero. 

Un esempio d'uso - MONK, classificazione - è in run_experiments_monk.ipynb.
Un altro esempio d'uso - CUP, regressione - è in run_experiments_cup.ipynb.

Cose rimaste da fare:
- Analisi. Questo include tutte le indicazioni di micheli su come variare gli iperparametri, quale validation schema usare, come fare model selection e come fare model assessment, eccetera - da leggere le indicazioni **molto chiaramente**. 
- Slides.
- Questo report è da tenere in considerazione (almeno per la parte di monk: https://github.com/dilettagoglia/impl-NN-from-scratch/blob/main/GOGLIA_MURGIA_Report.pdf)

MOLTO IMPORTANTE: PER QUALSIASI PROBLEMA CONSULTIAMOCI, IO SONO A DISPOSIZIONE 24/7 🔥 (ma non chiamarmi alle 4 di notte per verificare...)

# Cosa è stato fatto*:

\*: dovrebbe essere tutto ciò che micheli chiedeva per l'implementazione.

### Learning

- Learning di base

- Tecniche per il decadimento del learning rate. Parametri da passare al costruttore della rete:
    - lr_decay_type ("exponential", "linear", "none")
    - decay_rate (un float)

- Tecniche per il momenutm. Parametri da passare al costruttore della rete:
    - momentum_type ("none", "momentum", "nesterov momentum")
    - momentum_alpha (un float)

- Tecniche per l'inizializzazione dei pesi. Parametri da passare al costruttore della rete:
    - weight_init_type ("base", "glorot") // base usa il fan-in, glorot usa fan-in e fan-out

- Tecniche per la regolarizzazione. 
    - Parametri da passare al costruttore della rete:
        - reg_type ("l2", "l1", "none" // oppure lambda_reg=0)
        - lambda_reg (un float),
    - File di riferimento: lib/regularization.py

- Early stopping. Parametri da passare al metodo `train` della rete:
    - early_stopping (True, False)
    - patience (un int)
    - min_delta (un float)

- MiniBatch training, che astrae stochastic, full-batch

### Model Selection

- k-Fold cross validation, in lib/cross_validation.py (capisci meglio dai notebook di esempio)
- Grid search, parallela, con k-fold cross validation, in lib/grid_search.py


---