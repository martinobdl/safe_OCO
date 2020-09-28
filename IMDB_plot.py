import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


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
        'COGD': 'orange'
        }

for yaml_file in glob.glob('experiments/IMDB/*.yaml'):
    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

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
    if d['algo']['name'] == 'COGD' or d['algo']['name'] == 'DPOGD':
        beta[d['algo']['name']] = data['beta']

breakpoint()
plt.figure()
for k in R.keys():
    T = np.arange(len(R[k]))*d['checkpoints']
    plt.plot(T, R[k]/T, label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('Regret')
plt.show(block=False)

plt.figure()
for k in B.keys():
    T = np.arange(len(B[k]))*d['checkpoints']
    plt.plot(T, B[k], label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('bdgt')
plt.show(block=False)

plt.figure()
for k in beta.keys():
    T = np.arange(len(beta[k]))*d['checkpoints']
    plt.plot(T, beta[k], label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('beta')
plt.show(block=False)

plt.figure()
for k in L.keys():
    T = np.arange(len(L[k]))*d['checkpoints']
    plt.plot(T, L[k], label=k, color=colors[k])
plt.legend()
plt.title('Loss')
plt.grid(True)
plt.show()
