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
        'DPOGD': 'green',
        'COGD': 'orange',
        'ADAM': 'black',
        'DPOGDMAX': 'cyan',
        'ADAGRAD': 'blue'
        }

for yaml_file in glob.glob('experiments/SPAM/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))
    if d['algo']['name'] == 'ADAM' and abs(d['algo']['alpha'] - 0.001) < 1e-3:
        pass
    else:
        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        L_t = np.cumsum(data['loss_t'])
        L_def_t = np.cumsum(data['loss_def_t'])
        L_best = np.cumsum(data['best_loss_t'])
        alpha = 0.01
        bdg = (1 + alpha)*L_def_t - L_t
        R_t = L_t - L_best
        R[d['algo']['name']] = R_t
        B[d['algo']['name']] = bdg
        L[d['algo']['name']] = L_t
        if d['algo']['name'] in ['COGD', 'DPOGD', 'DPOGDMAX']:
            beta[d['algo']['name']] = data['beta']

plt.figure()
for k in R.keys():
    T = np.arange(1, len(R[k])+1)*d['checkpoints']
    idx = utils.range_to_idx(np.arange(1, len(T), 1000))
    T = T[idx]
    plt.plot(T, R[k][idx], label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('Regret')
plt.show(block=False)

tikzplotlib.save("teximgs/SPAM_regret.tex")

plt.figure()
for k in B.keys():
    T = np.arange(len(B[k]))*d['checkpoints']
    idx = utils.range_to_idx(np.arange(1, len(T)/4, 1000))
    T = T[idx]
    plt.plot(T, B[k][idx], label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('bdgt')
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