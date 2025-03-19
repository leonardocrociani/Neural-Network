"""
This script has been created to analyze the hyperparameters search results.
It generates plots to visualize the relationship between hyperparameters and the average metric.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "../results/hyperparams_analysis/"
os.makedirs(output_dir, exist_ok=True)

with open('../results/hyperparams_search/cup/fine/gs_results.json', 'r') as f:
    all_params = json.load(f)

data = []
for entry in all_params:
    row = {}
    row["average_metric"] = entry["average_metric"]
    row["std_metric"] = entry["std_metric"]
    for key, value in entry["params"].items():
        row[key] = value
    data.append(row)

df = pd.DataFrame(data)

numeric_columns = ["decay_rate", "lambda_reg", "learning_rate", "momentum_alpha", "batch_size", "average_metric", "std_metric"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_hyperparams = ["decay_rate", "lambda_reg", "learning_rate", "momentum_alpha", "batch_size"]
categorical_hyperparams = ["lr_decay_type", "momentum_type", "reg_type", "weight_init"]

for param in numeric_hyperparams:
    if param in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=param, y="average_metric")
        plt.title(f"{param} vs Average Metric")
        plt.xlabel(param)
        plt.ylabel("Average Metric")
        
        if param == "lambda_reg":
            plt.xscale("symlog", linthresh=1e-2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_vs_average_metric.png"))
        plt.close()

for param in categorical_hyperparams:
    if param in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=param, y="average_metric")
        plt.title(f"{param} vs Average Metric")
        plt.xlabel(param)
        plt.ylabel("Average Metric")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{param}_vs_average_metric.png"))
        plt.close()

if "layers" in df.columns:
    df["layers_str"] = df["layers"].apply(lambda x: "-".join(map(str, x)) if isinstance(x, list) else str(x))
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="layers_str", y="average_metric")
    plt.title("Layers Configuration vs Average Metric")
    plt.xlabel("Layers Configuration")
    plt.ylabel("Average Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layers_vs_average_metric.png"))
    plt.close()
