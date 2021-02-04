import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import src.utils as utils
import tikzplotlib


DP = []
C = []
OGD = []
R = {}
B = {}
L = {}
beta = {}

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

for yaml_file in glob.glob('experiments2/SPAM/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))
    base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
    np_file = os.path.join(os.path.dirname(yaml_file), base_np)
    data = dict(np.load(np_file))

    L_t = data['L_t'].T[0]
    L_def_t = data['LT_t'].T[0]
    L_best = data['LS_t'].T[0]
    alpha = 0.01
    bdg = (1 + alpha)*L_def_t - L_t
    R_t = L_t - L_best
    R[d['algo']['name']] = R_t
    B[d['algo']['name']] = bdg
    L[d['algo']['name']] = L_t
    if d['algo']['name'] in ['DPWRAP_OGD', 'CWRAP_OGD']:
        beta[d['algo']['name']] = data['beta']

keys = [k for k, _ in sorted(label.items(), key=lambda x: x[1])]

plt.figure()
for k in keys:
    T = np.arange(1, len(R[k])+1)*d['checkpoints']
    idx = utils.range_to_idx(np.arange(1, len(T), 10))
    T = T[idx]
    plt.plot(T, R[k][idx], label=label[k], color=colors[k], marker=marker[k], linestyle=linestyle[k], markevery=500, markersize=3)
plt.ylim(top=2100)
plt.xlim(right=T[-1]-30)
plt.xlabel(r"$t$")
plt.ylabel(r"$R_t$")
plt.show(block=False)

tikzplotlib.save("teximgs/SPAM_regret.tex")

plt.figure()
for k in keys:
    T = np.arange(len(B[k]))*d['checkpoints']
    idx = utils.range_to_idx(np.arange(1, 4000, 1))
    T = T[idx]
    plt.plot(T, B[k][idx], label=label[k], color=colors[k], marker=marker[k], linestyle=linestyle[k], markevery=400, markersize=3)
plt.xlim(right=T[-1]-30)
plt.legend()
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='k', linewidth=0.8)
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$Z_t$")
plt.show(block=False)

tikzplotlib.save("teximgs/SPAM_bdgt.tex")

plt.figure()
for k in beta.keys():
    T = np.arange(len(beta[k]))*d['checkpoints']
    plt.plot(T, beta[k], label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('beta')
plt.show()

