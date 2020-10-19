import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from src import utils
import tikzplotlib

# Regret

DP = []
C = []
OGD = []
D = defaultdict(lambda: [])

colors = {
        'OGD': 'red',
        'DPOGD': 'green',
        'COGD': 'orange',
        'DPOGDMAX': 'cyan',
        'ADAGRAD': 'blue'
        }

for yaml_file in glob.glob('experiments/OLR/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    if d['algo']['name'] != 'DPOGDMAX':
        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
        x0_true = utils.clean_dump_vector(d['env']['beta'])
        D_tilde = np.linalg.norm(x0_true-x0_safe)

        L_t = np.cumsum(data['loss_t'])
        L_best_t = np.cumsum(data['best_loss_t'])
        R_t = L_t - L_best_t
        if abs(D_tilde - 0.5) < 1e-3:
            D[d['algo']['name']].append(R_t)


label = {
        'OGD': 'OGD',
        'DPOGD': 'DPOGD',
        'COGD': 'COGD',
        'ADAGRAD': 'AdaGrad'
        }


plt.figure()
for k in D.keys():
    T = np.arange(len(D[k][0]))*d['checkpoints']
    idx = np.arange(0, len(T), 500)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, D[k], idx=idx, speed=1)
    plt.plot(T, Y, label=label[k], color=colors[k])
    plt.fill_between(T, LB, UB, alpha=0.2, color=colors[k])
plt.legend()
# plt.grid(True)
tikzplotlib.save("teximgs/OLR_regret.tex")
plt.ylabel(r"$R_t$")
plt.xlabel(r"$time$")
plt.show()
