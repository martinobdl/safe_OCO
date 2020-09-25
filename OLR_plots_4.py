import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from src import utils
import numpy as np


DPOGD1 = []
DPOGD2 = []
DPOGD3 = []
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
    D_tilde = np.linalg.norm(x0_true-x0_safe)

    L_t = np.cumsum(data['loss_t'])
    L_best_t = np.cumsum(data['best_loss_t'])
    R_t = L_t - L_best_t
    if abs(D_tilde - 0.3) < 1e-3:
        if d['algo']['name'] == 'OGD':
            OGD1.append(L_t)
        if d['algo']['name'] == 'DPOGD':
            DPOGD1.append(L_t)
    if abs(D_tilde - 0.5) < 1e-3:
        if d['algo']['name'] == 'OGD':
            OGD2.append(L_t)
        if d['algo']['name'] == 'DPOGD':
            DPOGD2.append(L_t)
    if abs(D_tilde - 0.9) < 1e-3:
        if d['algo']['name'] == 'OGD':
            OGD3.append(L_t)
        if d['algo']['name'] == 'DPOGD':
            DPOGD3.append(L_t)

OGD1 = np.array(OGD1)
OGD2 = np.array(OGD2)
OGD3 = np.array(OGD3)
DPOGD1 = np.array(DPOGD1)
DPOGD2 = np.array(DPOGD2)
DPOGD3 = np.array(DPOGD3)

T = np.arange(0, d['env']['T'], d['checkpoints'])
plt.figure()
plt.plot(T, np.mean(DPOGD1-OGD1, axis=0), label=r"$\tilde D=0.3$")
plt.plot(T, np.mean(DPOGD2-OGD2, axis=0), label=r"$\tilde D=0.5$")
plt.plot(T, np.mean(DPOGD3-OGD3, axis=0), label=r"$\tilde D=0.7$")
plt.legend()
plt.ylabel(r'$L_t(DPOGD)-L_t(ODG)$')
plt.grid(True)
plt.show()
