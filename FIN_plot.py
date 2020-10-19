import yaml
import tikzplotlib
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


DP = []
C = []
OGD = []
W = {}
B = {}
L = {}
beta = {}

colors = {
        'OGD': 'red',
        'DPOGD': 'green',
        'COGD': 'orange',
        'ADAM': 'black',
        }

for yaml_file in glob.glob('experiments/FIN/*.yaml'):

    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    if d['algo']['name'] != 'DPOGDMAX':
        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        L_t = np.cumsum(data['loss_t'])
        L_def_t = np.cumsum(data['loss_def_t'])
        # L_best = np.cumsum(data['best_loss_t'])
        b = 0.80
        u = 1.04
        ll = 0.96
        e_u = float(np.log(u)-np.log(ll))
        alpha = float(b/(43140*e_u))
        print(b)
        bdg = (1 + alpha)*L_def_t - L_t
        t = np.arange(1, len(L_t)+1)
        K = -np.log(ll)
        W_t = np.exp(-L_t+K*t)
        D2 = np.exp(-L_def_t+K*t)
        W[d['algo']['name']] = W_t
        B[d['algo']['name']] = W_t-D2*b
        L[d['algo']['name']] = L_t
        D = D2*b
        if d['algo']['name'] in ['COGD', 'DPOGD', 'DPOGDMAX']:
            beta[d['algo']['name']] = data['beta']

plt.figure()
for k in W.keys():
    T = np.arange(1, len(W[k])+1)*d['checkpoints']
    idx = np.arange(1, len(T)+1, 100)
    T = T[idx]
    plt.plot(T, W[k][idx], label=k, color=colors[k])
# plt.plot(T, D, label='def*b', color='blue')
# plt.plot(T, D2, label='def', color='magenta')
plt.legend()
# plt.grid(True)
# plt.title('W')
plt.show(block=False)

tikzplotlib.save("teximgs/FIN_wealth.tex")

plt.figure()
for k in B.keys():
    T = np.arange(len(B[k]))*d['checkpoints']
    T = T[idx]
    plt.plot(T, B[k][idx], label=k, color=colors[k])
plt.legend()
# plt.grid(True)
plt.title('bdgt')
plt.show(block=False)

tikzplotlib.save("teximgs/FIN_budget.tex")

plt.figure()
for k in beta.keys():
    T = np.arange(len(beta[k]))*d['checkpoints']
    plt.plot(T, beta[k], label=k, color=colors[k])
plt.legend()
plt.grid(True)
plt.title('beta')
plt.show()
