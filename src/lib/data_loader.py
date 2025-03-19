"""
This module contains functions to load the datasets, and to save the final predictions.
"""

import os
import pandas as pd
from urllib.request import urlretrieve
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import csv

TEAM_NAME = 'LambdaX'
CACHE_DIR = "../datasets/monks"
os.makedirs(CACHE_DIR, exist_ok=True)

monk_urls = {
    "MONK-1": {
        "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train",
        "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test",
    },
    "MONK-2": {
        "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train",
        "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test",
    },
    "MONK-3": {
        "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train",
        "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test",
    },
}

def download_if_not_exists(url, filepath):
    """
    Download the file from the given url if it does not exist in the given filepath.
    Args:
        url: str, the url of the file to download.
        filepath: str, the path where to save the file.
    """
    if not os.path.exists(filepath):
        print(f"Downloading {filepath}...")
        urlretrieve(url, filepath)
    else:
        print(f"Using cached {filepath}")

def shuffle_df(df):
    """
    Shuffle the dataframe rows.
    Args:
        df: pd.DataFrame, the dataframe to shuffle.
    Returns:
        pd.DataFrame, the shuffled dataframe
    """
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def postprocess_label(y):
    """
    Postprocess the label series.
    [NOTE: It transforms the label series into a 2D numpy array, because the fit function of sklearn requires a 2D array.]
    Args:
        y: pd.Series, the label series.
    Returns:
        np.ndarray, the postprocessed label array.
    """
    y = pd.Series(y).to_numpy()
    y = np.expand_dims(y, 1)
    return y

def get_monks_dataset(id, one_hot_encode=False):
    """
    Load the MONK dataset with the given id.
    Args:
        id: int, the id of the MONK dataset.
        one_hot_encode: bool, whether to one-hot encode the dataset.
    Returns:
        X_train, y_train, X_test, y_test: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    columns = ["label", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    encoder = OneHotEncoder()
    name = f"MONK-{id}"
    urls = monk_urls[name]

    train_path = os.path.join(CACHE_DIR, f"{name.lower()}-train.csv")
    test_path = os.path.join(CACHE_DIR, f"{name.lower()}-test.csv")

    download_if_not_exists(urls["train"], train_path)
    download_if_not_exists(urls["test"], test_path)

    df_train = pd.read_csv(train_path, delimiter=" ", names=columns)
    df_train.set_index("id", inplace=True)
    df_train = shuffle_df(df_train) # mescola le righe prima dello split, ovviamente.
    
    y_train = df_train["label"]
    y_train = postprocess_label(y_train)
    X_train = df_train.drop(columns=["label"])

    df_test = pd.read_csv(test_path, delimiter=" ", names=columns)
    df_test.set_index("id", inplace=True)
    df_test = shuffle_df(df_test)
    
    y_test = df_test["label"]
    y_test = postprocess_label(y_test)
    X_test = df_test.drop(columns=["label"])

    if one_hot_encode:
        print(f"One-hot encoding {name} dataset...")
        X_train_encoded = encoder.fit_transform(X_train).toarray()
        X_test_encoded = encoder.transform(X_test).toarray()

        return X_train_encoded, y_train, X_test_encoded, y_test

    return X_train.values, y_train, X_test.values, y_test

def get_cup_dataset(dev_set_size=0.8):
    """
    Return the dev set [TR + VL] and the test set [TS]. The output to submit must be done on a different test set: use get_cup_final_test_set().
    The sets are returned in this order: X_dev, y_dev, X_test, y_test
    Args:
        dev_set_size: float, the size of the dev set.
    Returns:
        X_dev, y_dev, X_test, y_test: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    column_names = ["ID"] + [f"INPUT_{i+1}" for i in range(12)] + ["TARGET_x", "TARGET_y", "TARGET_z"]
    
    df = pd.read_csv("../datasets/cup/ML-CUP24-TR.csv", comment='#', header=None, names=column_names)
    
    X = df.drop(columns=["ID", "TARGET_x", "TARGET_y", "TARGET_z"]).to_numpy()
    y = df[["TARGET_x", "TARGET_y", "TARGET_z"]].to_numpy()

    N = X.shape[0] # dataset size
    N_dev = int(N * dev_set_size)
    X_dev, y_dev = X[:N_dev], y[:N_dev]
    X_test, y_test = X[N_dev:], y[N_dev:]
    
    return X_dev, y_dev, X_test, y_test


def get_cup_final_test_set():
    """
    Return the final test set [TS] to submit the predictions.
    Returns:
        X: np.ndarray
    """
    column_names = ["ID"] + [f"INPUT_{i+1}" for i in range(12)]
    df = pd.read_csv("../datasets/cup/ML-CUP24-TS.csv", comment='#', header=None, names=column_names)
    X = df.drop(columns=["ID"]).to_numpy()
    return X

def save_final_prediction(y_pred):
    """
    Save the final predictions on the test set.
    Args:
        y_pred: np.ndarray, the predictions to save.
    """
    with open(f"../results/blind_test/{TEAM_NAME}_ML-CUP24-TS.csv", "w", newline="\n") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['# Leonardo Crociani'])
        writer.writerow([f'# {TEAM_NAME}'])
        writer.writerow(['# ML-CUP24 v1'])
        writer.writerow(['# 20 March 2025'])
        for i in range(len(y_pred)):
            writer.writerow([str(i+1)]+list(y_pred[i]))
            
    print('\n# ==== Final predictions saved. ==== #\n')