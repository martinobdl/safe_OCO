import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from src import utils
from collections import defaultdict
import matplotlib.cm as cm
import tikzplotlib

cmap = cm.tab20b

CPOGD = defaultdict(lambda: [])
CSOGD = defaultdict(lambda: [])
OGD = defaultdict(lambda: [])
ADAGRAD = defaultdict(lambda: [])
R2 = defaultdict(lambda: [])

colors = {
        'OGD': 'red',
        'DPWRAP_OGD': 'green',
        'CWRAP_OGD': 'orange',
        'ADAGRAD': 'blue',
        'R2': 'deeppink'
        }

label = {
        'OGD': 'OGD',
        'DPWRAP_OGD': 'CP-OGD',
        'CWRAP_OGD': 'CS-OGD',
        'ADAGRAD': 'Adagrad',
        'R2': 'CRDG'
        }

marker = {
        'OGD': '.',
        'DPWRAP_OGD': '>',
        'CWRAP_OGD': '+',
        'ADAGRAD': '*',
        'R2': 'p'
        }

linestyle = {
        'OGD': 'solid',
        'DPWRAP_OGD': 'dotted',
        'CWRAP_OGD': 'dashed',
        'ADAGRAD': 'dashdot',
        'R2': 'solid'
        }

for yaml_file in glob.glob('experiments2/OLR2/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
    np_file = os.path.join(os.path.dirname(yaml_file), base_np)
    data = dict(np.load(np_file))

    x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
    x0_true = utils.clean_dump_vector(d['env']['beta'])
    D_tilde = np.round(np.linalg.norm(x0_true-x0_safe), 1)

    L_t = data['L_t']
    L_best_t = data['LS_t']
    T = int(10000/d['checkpoints'])
    r = (L_t[T] - L_best_t[T])[0]
    if d['algo']['name'] == 'OGD':
        OGD[D_tilde].append(r)
    if d['algo']['name'] == 'DPWRAP_OGD':
        CPOGD[D_tilde].append(r)
    if d['algo']['name'] == 'CWRAP_OGD':
        CSOGD[D_tilde].append(r)
    if d['algo']['name'] == 'ADAGRAD':
        ADAGRAD[D_tilde].append(r)
    if d['algo']['name'] == 'ConversionConstrained_RewardDoubligNDGuess':
        R2[D_tilde].append(r)
        # pass

    tot = {
            'OGD': OGD,
            'DPWRAP_OGD': CPOGD,
            'CWRAP_OGD': CSOGD,
            'ADAGRAD': ADAGRAD,
            'R2': R2
            }

# breakpoint()
keys = [k for k, _ in sorted(label.items(), key=lambda x: x[1])]

plt.figure()
for k in keys:
    T = list(tot[k].keys())
    T.sort()

    A = []
    for s in range(len(tot[k][1])):
        tmp = []
        for d in np.arange(0.5, 4, 0.5):
            tmp.append(tot[k][d][s])
        A.append(np.array(tmp))

    tot[k] = list(tot[k].values())
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, A, speed=1)
    if k in ['OGD', 'ADAGRAD']:
        Y = np.mean(Y)*np.ones_like(Y)
        LB = np.mean(LB)*np.ones_like(LB)
        UB = np.mean(UB)*np.ones_like(UB)
    plt.plot(T, Y, label=label[k], color=colors[k], marker=marker[k], markevery=1, linestyle=linestyle[k], markersize=3)
    plt.fill_between(T, LB, UB, alpha=0.2, color=colors[k])
plt.ylim(top=10000)
plt.ylim(bottom=-100)
plt.legend()
plt.ylabel(r'$R_T$')
plt.xlabel(r'$\tilde {D}$')
# plt.grid(True)

tikzplotlib.save("teximgs/OLR_D.tex")
plt.show()
