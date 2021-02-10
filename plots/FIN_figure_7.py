import yaml
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
        'CP_OGD': 'green',
        'CS_OGD': 'orange',
        'ADAGRAD': 'blue',
        'CRDG': 'deeppink'
        }

label = {
        'OGD': 'OGD',
        'CP_OGD': 'CP-OGD',
        'CS_OGD': 'CS-OGD',
        'ADAGRAD': 'Adagrad',
        'CRDG': 'CRDG'
        }

marker = {
        'OGD': '.',
        'CP_OGD': '>',
        'CS_OGD': '+',
        'ADAGRAD': '*',
        'CRDG': 'p'
        }

linestyle = {
        'OGD': 'solid',
        'CP_OGD': 'dotted',
        'CS_OGD': 'dashed',
        'ADAGRAD': 'dashdot',
        'CRDG': 'solid'
        }

for yaml_file in glob.glob('experiments/FIN/*.yaml'):

    with open(yaml_file, 'r') as f:
        d = dict(yaml.load(f, Loader=yaml.FullLoader))

    base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
    np_file = os.path.join(os.path.dirname(yaml_file), base_np)
    data = dict(np.load(np_file))

    L_t = data['L_t'].T[0]
    L_def_t = data['LT_t'].T[0]
    b = 0.80
    u = 1.04
    ll = 0.96
    e_u = float(np.log(u)-np.log(ll))
    alpha = float(b/(43140*e_u))
    bdg = (1 + alpha)*L_def_t - L_t
    t = np.arange(1, len(L_t)+1)
    K = -np.log(ll)
    W_t = np.exp(-L_t+K*t)
    D2 = np.exp(-L_def_t+K*t)
    W[d['algo']['name']] = W_t
    B[d['algo']['name']] = W_t-D2*b
    L[d['algo']['name']] = L_t
    D = D2*b

plt.figure()
for k in W.keys():
    T = np.arange(1, len(W[k])+1)*d['checkpoints']
    idx = np.arange(1, len(T)+1, 100)
    T = T[idx]
    plt.plot(T, W[k][idx], color=colors[k], label=label[k], marker=marker[k],
             linestyle=linestyle[k], markevery=50, markersize=8)
plt.legend()
plt.title('Fig7(a) Wealth')
plt.xlabel(r"$t$")
plt.ylabel(r"$W_t$")
plt.show(block=False)


plt.figure()
for k in B.keys():
    T = np.arange(len(B[k]))*d['checkpoints']
    T = T[idx]
    plt.plot(T, B[k][idx], color=colors[k], label=label[k], marker=marker[k],
             linestyle=linestyle[k], markevery=50, markersize=8)
plt.title('Fig7(b) Wealth Budget')
plt.xlabel(r"$t$")
plt.ylabel(r"$P_t$")
plt.legend()
plt.show()
