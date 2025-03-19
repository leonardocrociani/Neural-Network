"""
[Less Relevant Code]
This script just monitors a grid search process, sending a telegram message to me as soon as the process is done.
"""

import subprocess
import time
import json
from lib.grid_search import get_best_results
from lib.tg import send_tg_msg, send_tg_img

while True:
    out = subprocess.check_output('ps xa | grep joblib | wc -l', shell=True)
    out = out.decode('utf-8').strip()
    print(out)
    try:
        out = int(out)
        if out < 12:
            raise Exception('DONE')
        time.sleep(5)
    except:
        send_tg_msg('GS terminata!')
        break

with open("../results/hyperparams_search/monks_3_reg/gs_results.json", "r") as f:
    params = json.load(f)
    
params = get_best_results(params, strategy="max_accuracy_min_metric_std")
send_tg_msg(json.dumps(params, indent=4))

send_tg_msg('Model assessment...')

time.sleep(5)

out = subprocess.check_output('source ../venv/bin/activate && python model_assessment_monks.py 3_reg', shell=True)

loss_path = '../results/hyperparams_search/monks_3_reg/loss.png'
accuracy_path = '../results/hyperparams_search/monks_3_reg/accuracy.png'

send_tg_msg('Loss:')
send_tg_img(loss_path)
send_tg_msg('Accuracy:')
send_tg_img(accuracy_path)
send_tg_msg(out)