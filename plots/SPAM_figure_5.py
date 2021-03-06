import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import src.utils as utils


DP = []
C = []
OGD = []
R = {}
B = {}
L = {}
beta = {}

colors = {
        'OGD': 'red',
        'CP_OGD': 'green',
        'CS_OGD': 'orange',
        'ADAGRAD': 'blue',
        'CRDG': 'deeppink'
        }

label = {
        'OGD': 'OGD',
        'CP_OGD': 'CP-OGD',
        'CS_OGD': 'CS-OGD',
        'ADAGRAD': 'Adagrad',
        'CRDG': 'CRDG'
        }

marker = {
        'OGD': '.',
        'CP_OGD': '>',
        'CS_OGD': '+',
        'ADAGRAD': '*',
        'CRDG': 'p'
        }

linestyle = {
        'OGD': 'solid',
        'CP_OGD': 'dotted',
        'CS_OGD': 'dashed',
        'ADAGRAD': 'dashdot',
        'CRDG': 'solid'
        }

for yaml_file in glob.glob('experiments/SPAM/*.yaml'):
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
    if d['algo']['name'] in ['CP_OGD', 'CS_OGD']:
        beta[d['algo']['name']] = data['beta']

keys = [k for k, _ in sorted(label.items(), key=lambda x: x[1])]

plt.figure()
for k in keys:
    T = np.arange(1, len(R[k])+1)*d['checkpoints']
    idx = utils.range_to_idx(np.arange(1, len(T), 10))
    T = T[idx]
    plt.plot(T, R[k][idx], label=label[k], color=colors[k], marker=marker[k],
             linestyle=linestyle[k], markevery=500, markersize=8)
plt.ylim(top=2100)
plt.legend()
plt.title('Fig5(a) SPAM Regret')
plt.xlim(right=T[-1]-30)
plt.xlabel(r"$t$")
plt.ylabel(r"$R_t$")
plt.show(block=False)


plt.figure()
for k in keys:
    T = np.arange(len(B[k]))*d['checkpoints']
    idx = utils.range_to_idx(np.arange(1, 4000, 1))
    T = T[idx]
    plt.plot(T, B[k][idx], label=label[k], color=colors[k], marker=marker[k],
             linestyle=linestyle[k], markevery=400, markersize=8)
plt.xlim(right=T[-1]-30)
plt.title('Fig5(b) SPAM Budget')
plt.legend()
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='k', linewidth=0.8)
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$Z_t$")
plt.show()
