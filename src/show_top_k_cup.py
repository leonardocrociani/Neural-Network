"""
[Less Relevant Code]
This script was implemented to show the top k hyperparameters from the grid search results for the cup.
"""

from sklearn.model_selection import train_test_split
from lib.data_loader import get_cup_dataset
from lib.utils import parse_gs_type
from lib.grid_search import sort_gs_results
import json

COARSE_OR_FINE = parse_gs_type() # parse <coarse|fine> from cli
SHOW_TOP_K=10
SHOW_FULL_OR_TOP_PARAMS = 'top-params' # or 'full'

X_dev, y_dev, X_test, y_test = get_cup_dataset()
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=42
)

with open(f"../results/hyperparams_search/cup/{COARSE_OR_FINE}/gs_results.json", "r") as f:
    params = json.load(f)

params = sort_gs_results(params)[:SHOW_TOP_K]


params_count = {
    'learning_rate': {},
    'layers': {},
    'decay_rate': {},
    'lambda_reg': {},
    'momentum_alpha': {},
    'momentum_type': {},
    'weight_init': {},
    'activation_function_names': {},
    'reg_type': {},
    'lr_decay_type': {},
    'batch_size' : {}
}

for idx, param in enumerate(params):
    p = param['params']
    
    if SHOW_FULL_OR_TOP_PARAMS == 'full':
        print(json.dumps(p, indent=4))
        print('\n\n')
    
    weight = 1 / (idx + 1)

    params_count['learning_rate'][p['learning_rate']] = params_count['learning_rate'].get(p['learning_rate'], 0) + weight
    params_count['layers'][str(p['layers'])] = params_count['layers'].get(str(p['layers']), 0) + weight
    params_count['decay_rate'][p['decay_rate']] = params_count['decay_rate'].get(p['decay_rate'], 0) + weight
    params_count['lambda_reg'][p['lambda_reg']] = params_count['lambda_reg'].get(p['lambda_reg'], 0) + weight
    params_count['momentum_alpha'][p['momentum_alpha']] = params_count['momentum_alpha'].get(p['momentum_alpha'], 0) + weight
    params_count['momentum_type'][p['momentum_type']] = params_count['momentum_type'].get(p['momentum_type'], 0) + weight
    params_count['weight_init'][p['weight_init']] = params_count['weight_init'].get(p['weight_init'], 0) + weight
    params_count['activation_function_names'][str(p['activation_function_names'])] = params_count['activation_function_names'].get(str(p['activation_function_names']), 0) + weight
    params_count['reg_type'][p['reg_type']] = params_count['reg_type'].get(p['reg_type'], 0) + weight
    params_count['lr_decay_type'][p['lr_decay_type']] = params_count['lr_decay_type'].get(p['lr_decay_type'], 0) + weight
    params_count['batch_size'][p['batch_size']] = params_count['batch_size'].get(p['batch_size'], 0) + weight

if SHOW_FULL_OR_TOP_PARAMS == 'top-params':
    for key in params_count:
        total_weight = sum(params_count[key].values())
        for sub_key in params_count[key]:
            params_count[key][sub_key] /= total_weight

    print(json.dumps(params_count, indent=4))