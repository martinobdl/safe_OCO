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

DPOGD = defaultdict(lambda: [])
OGD = defaultdict(lambda: [])
DPOGD1 = []
DPOGD2 = []
DPOGD3 = []
DPOGDMAX1 = []
DPOGDMAX2 = []
DPOGDMAX3 = []
OGD1 = []
OGD2 = []
OGD3 = []

colors = {
        'OGD': 'red',
        'DPOGD': 'green',
        'COGD': 'orange'
        }

for yaml_file in glob.glob('experiments/OLR/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
    np_file = os.path.join(os.path.dirname(yaml_file), base_np)
    data = dict(np.load(np_file))

    x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
    x0_true = utils.clean_dump_vector(d['env']['beta'])
    D_tilde = np.round(np.linalg.norm(x0_true-x0_safe), 1)

    L_t = np.cumsum(data['loss_t'])
    L_best_t = np.cumsum(data['best_loss_t'])
    if d['algo']['name'] == 'OGD':
        OGD[D_tilde].append(L_t)
    if d['algo']['name'] == 'DPOGD':
        DPOGD[D_tilde].append(L_t)

D = defaultdict(lambda: [])
for k in OGD.keys():
    for j1, j2 in zip(OGD[k], DPOGD[k]):
        D[k].append(j2-j1)


def color(D_tilde):
    return cmap((D_tilde)/1.3)


plt.figure()
for k in range(1, 13)[::-1]:
    k = np.round(k/10, 1)
    T = np.arange(len(D[k][0]))*d['checkpoints']
    idx = np.arange(0, len(T), 100)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, D[k], idx=idx, speed=1)
    plt.plot(T, Y, label=r"$\tilde D={}$".format(k), color=color(k))
    plt.fill_between(T, LB, UB, alpha=0.2, color=color(k))
plt.legend()
plt.ylabel(r'$L_t(DPOGD, ODG)$')
plt.xlabel(r'$t$')
# plt.grid(True)

tikzplotlib.save("teximgs/OLR_D.tex")
plt.show()