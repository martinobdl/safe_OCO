import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from src import utils
from collections import defaultdict
import matplotlib.cm as cm

cmap = cm.tab20b

CP = defaultdict(lambda: [])
CS = defaultdict(lambda: [])

colors = {
        'CP_OGD': 'green',
        'CS_OGD': 'orange'
        }

for yaml_file in glob.glob('experiments2/OLR/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    if d['algo']['name'] in colors.keys():
        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
        x0_true = utils.clean_dump_vector(d['env']['beta'])
        D_tilde = np.round(np.linalg.norm(x0_true-x0_safe), 1)

        r_t = (40*2)**0.5*(1-np.minimum(1, np.maximum(0, data['beta'].ravel())))
        if d['algo']['name'] == 'CS_OGD':
            CS[D_tilde].append(r_t)
        if d['algo']['name'] == 'CP_OGD':
            CP[D_tilde].append(r_t)

        assert np.max(r_t) <= (2*40)**0.5
        assert np.min(r_t) >= 0


def color(D_tilde):
    return cmap((D_tilde)/4)


D = [0.5, 1, 1.5, 2][::-1]

plt.figure()
for k in D:
    T = np.arange(len(CS[k][0]))*d['checkpoints']
    idx = 1 + (len(T)-2)/2**np.arange(0, 20, 0.1)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, CS[k], speed=1, idx=idx)
    LB = utils.clip(LB, 0, 80**0.5)
    UB = utils.clip(UB, 0, 80**0.5)
    plt.plot(T, Y, label=r"$\tilde D={}$".format(k), color=color(k))
    plt.xscale('log')
    plt.fill_between(T, LB, UB, alpha=0.2, color=color(k))
plt.legend()
plt.ylabel(r'$r_t(CS-OGD)$')
plt.xlabel(r'$t$')
# plt.grid(True)

plt.show(block=False)

plt.figure()
for k in D:
    T = np.arange(len(CP[k][0]))*d['checkpoints']
    idx = 1 + (len(T)-2)/2**np.arange(0, 20, 0.1)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, CP[k], speed=1, idx=idx)
    LB = utils.clip(LB, 0, 80**0.5)
    UB = utils.clip(UB, 0, 80**0.5)
    plt.plot(T, Y, label=r"$\tilde D={}$".format(k), color=color(k))
    plt.xscale('log')
    plt.fill_between(T, LB, UB, alpha=0.2, color=color(k))
plt.legend()
plt.ylabel(r'$r_t(CP-OGD)$')
plt.xlabel(r'$t$')
# plt.grid(True)

plt.show()
