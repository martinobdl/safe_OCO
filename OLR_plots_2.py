import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from src import utils


DP = []
C = []
OGD = []
D = defaultdict(lambda: [])

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
    L_def_t = np.cumsum(data['loss_def_t'])
    alpha = 0.05
    bdg = (1 + alpha)*L_def_t - L_t
    if abs(D_tilde - 0.3) < 1e-3:
        D[d['algo']['name']].append(bdg)


plt.figure()
for k in D.keys():
    Y = np.zeros_like(D[k][0])
    it = 0
    for y in D[k]:
        it += 1
        Y += np.array(y)
    Y = Y/it
    plt.plot(np.arange(len(Y))*d['checkpoints'], Y, label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.show()
