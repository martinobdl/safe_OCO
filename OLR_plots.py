import yaml
import glob
from src import utils
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

D_DP = []
D_C = []
R_DP = defaultdict(lambda: [])
T_DP = defaultdict(lambda: [])
R_C = defaultdict(lambda: [])
T_C = defaultdict(lambda: [])

for yaml_file in glob.glob('experiments/OLR/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    if (d['algo']['name'] == 'COGD' or d['algo']['name'] == 'DPOGD'):
        x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
        x0_true = utils.clean_dump_vector(d['env']['beta'])
        D_tilde = np.linalg.norm(x0_true-x0_safe)

        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        L_t_best = np.cumsum(data['best_loss_t'])
        L_t_DPOGD = np.cumsum(data['loss_t'])
        tau = np.argmin(data['beta'] > 1e-3)*d['checkpoints']
        R_T = L_t_DPOGD[-1] - L_t_best[-1]

        if d['algo']['name'] == 'DPOGD':
            D_DP.append(D_tilde)
            R_DP[float(np.round(D_tilde, 1))].append(R_T)
            T_DP[float(np.round(D_tilde, 1))].append(tau)
        else:
            D_C.append(D_tilde)
            R_C[float(np.round(D_tilde, 1))].append(R_T)
            T_C[float(np.round(D_tilde, 1))].append(tau)


def sort_and_plot(d, label):
    x = []
    y = []
    lb = []
    ub = []
    for k, v in d.items():
        x.append(k)
        y_m, l_m, u_m = utils.compute_mean_and_CI_bstr(np.array(v))
        y.append(y_m)
        lb.append(l_m)
        ub.append(u_m)
    y = [t for _, t in sorted(zip(x, y))]
    lb = [t for _, t in sorted(zip(x, lb))]
    ub = [t for _, t in sorted(zip(x, ub))]
    x.sort()
    plt.plot(x, y, '*-', label=label)
    plt.fill_between(x, lb, ub, alpha=0.2)


plt.figure()
sort_and_plot(R_DP, label='DPOGD')
sort_and_plot(R_C, label='COGD')
plt.xlabel('D')
plt.ylabel('R')
plt.legend()
plt.show(block=False)

plt.figure()
sort_and_plot(T_DP, label='DPOGD')
sort_and_plot(T_C, label='COGD')
plt.xlabel(r'$\tilde D$')
plt.ylabel(r'$\tau$')
plt.legend()
plt.show()
