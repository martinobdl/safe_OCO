import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from src import utils
import tikzplotlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-D', type=float, default=1)
parser.add_argument('-s', type=int, default=3)
args = parser.parse_args()
# Budget

DP = []
C = []
OGD = []
D = defaultdict(lambda: [])

colors = {
        'OGD': 'red',
        'DPWRAP_OGD': 'green',
        'CWRAP_OGD': 'orange',
        'ADAGRAD': 'blue',
        'ConversionConstrained_RewardDoubligNDGuess': 'deeppink'
        }

label = {
        'OGD': 'OGD',
        'DPWRAP_OGD': 'CP-OGD',
        'CWRAP_OGD': 'CS-OGD',
        'ADAGRAD': 'Adagrad',
        'ConversionConstrained_RewardDoubligNDGuess': 'CRDG'
        }

marker = {
        'OGD': '.',
        'DPWRAP_OGD': '>',
        'CWRAP_OGD': '+',
        'ADAGRAD': '*',
        'ConversionConstrained_RewardDoubligNDGuess': 'p'
        }

linestyle = {
        'OGD': 'solid',
        'DPWRAP_OGD': 'dotted',
        'CWRAP_OGD': 'dashed',
        'ADAGRAD': 'dashdot',
        'ConversionConstrained_RewardDoubligNDGuess': 'solid'
        }

for yaml_file in glob.glob('experiments2/OLR2/*.yaml'):
    try:
        with open(yaml_file, 'r') as f:
            d = dict(yaml.load(f, Loader=yaml.FullLoader))

        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
        x0_true = utils.clean_dump_vector(d['env']['beta'])
        D_tilde = np.linalg.norm(x0_true-x0_safe)

        L_t = data['L_t'].T[0]
        L_def_t = data['LT_t'].T[0]
        alpha = 0.01
        bdg = (1 + alpha)*L_def_t - L_t
        name = d['algo']['name']
        if abs(D_tilde - args.D) < 1e-3:
            D[name].append(bdg)
    except:
        print(yaml_file)
        pass

keys = [k for k, _ in sorted(label.items(), key=lambda x: x[1])]

plt.figure()
for k in keys:
    T = np.arange(len(D[k][0]))*d['checkpoints']
    idx = np.arange(0, 1050, 10)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, D[k], idx=idx, speed=args.s)
    plt.plot(T, Y, label=label[k], color=colors[k], marker=marker[k], markevery=10, linestyle=linestyle[k], markersize=3)
    plt.fill_between(T, LB, UB, alpha=0.2, color=colors[k])
plt.legend()
plt.xlim(right=T[-1]-30)
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='k', linewidth=0.8)
plt.xlabel(r"$t$")
plt.ylabel(r"$Z_t$")
plt.tight_layout()
tikzplotlib.save("teximgs/OLR_bdgt.tex")
plt.show()
