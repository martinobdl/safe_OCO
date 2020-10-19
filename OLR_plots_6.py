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
COGD = defaultdict(lambda: [])

colors = {
        'OGD': 'red',
        'DPOGD': 'green',
        'COGD': 'orange'
        }

for yaml_file in glob.glob('experiments/OLR/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    if d['algo']['name'] in ['COGD', 'DPOGD']:
        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
        x0_true = utils.clean_dump_vector(d['env']['beta'])
        D_tilde = np.round(np.linalg.norm(x0_true-x0_safe), 1)

        r_t = 2**0.5*(1-np.minimum(1, np.maximum(0, data['beta'].ravel())))
        if d['algo']['name'] == 'COGD':
            COGD[D_tilde].append(r_t)
        if d['algo']['name'] == 'DPOGD':
            DPOGD[D_tilde].append(r_t)

        assert np.max(r_t) <= 2**0.5
        assert np.min(r_t) >= 0


def color(D_tilde):
    return cmap((D_tilde)/1.3)


plt.figure()
for k in range(1, 13)[::-1]:
    k = np.round(k/10, 1)
    T = np.arange(len(COGD[k][0]))*d['checkpoints']
    idx = 1 + (len(T)-2)/2**np.arange(0, 20, 0.1)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, COGD[k], speed=1, idx=idx)
    LB = utils.clip(LB, 0, 2**0.5)
    UB = utils.clip(UB, 0, 2**0.5)
    plt.plot(T, Y, label=k, color=color(k))
    plt.xscale('log')
    plt.fill_between(T, LB, UB, alpha=0.2, color=color(k))
plt.legend()
plt.ylabel(r'$r_t(COGD)$')
plt.title(r'$COGD$')
plt.grid(True)

tikzplotlib.save("teximgs/OLR_r_COGD.tex")
plt.show(block=False)

plt.figure()
for k in range(1, 13)[::-1]:
    k = np.round(k/10, 1)
    T = np.arange(len(DPOGD[k][0]))*d['checkpoints']
    idx = 1 + (len(T)-2)/2**np.arange(0, 20, 0.1)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, DPOGD[k], idx=idx, speed=1)
    plt.plot(T, Y, label=k, color=color(k))
    plt.xscale('log')
    plt.fill_between(T, LB, UB, alpha=0.2, color=color(k))
plt.legend()
plt.ylabel(r'$r_t(DPOGD)$')
plt.title(r'$DPOGD$')
plt.grid(True)

tikzplotlib.save("teximgs/OLR_r_DPOGD.tex")
plt.show()
