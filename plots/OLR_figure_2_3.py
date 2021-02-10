import yaml
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from src import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-D', type=float, default=0.5)
parser.add_argument('-s', type=int, default=1)
args = parser.parse_args()

L = defaultdict(lambda: {})
R = defaultdict(lambda: [])
D = defaultdict(lambda: [])
CPOGD = defaultdict(lambda: [])
CSOGD = defaultdict(lambda: [])
OGD = defaultdict(lambda: [])
ADAGRAD = defaultdict(lambda: [])
CRDG = defaultdict(lambda: [])

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

for yaml_file in glob.glob('experiments/OLR/*.yaml'):
    try:
        with open(yaml_file, 'r') as f:
            d = dict(yaml.load(f, Loader=yaml.FullLoader))

        base_np = os.path.splitext(os.path.basename(yaml_file))[0]+'.npz'
        np_file = os.path.join(os.path.dirname(yaml_file), base_np)
        data = dict(np.load(np_file))

        x0_safe = utils.clean_dump_vector(d['env']['safe_strategy']['x0'])
        x0_true = utils.clean_dump_vector(d['env']['beta'])
        D_tilde = np.linalg.norm(x0_true-x0_safe)

        L_t = data['L_t'].T[0]
        L_def_t = data['LT_t'].T[0]
        LS_t = data['LS_t'].T[0]
        alpha = 0.01
        bdg = (1 + alpha)*L_def_t - L_t
        reg = L_t - LS_t
        T = int(10000/d['checkpoints'])
        r = (data['L_t'][T] - data['LS_t'][T])[0]
        name = d['algo']['name']
        seed = d['env']['seed']
        if abs(D_tilde - args.D) < 1e-3 or name in ['ADAGRAD', 'OGD']:
            L[name][seed] = L_t
            R[name].append(reg)
            if name not in ['ADAGRAD', 'OGD']:
                D[seed] = L_def_t

        D_tilde = np.round(np.linalg.norm(x0_true-x0_safe), 1)
        if d['algo']['name'] == 'OGD':
            for D_tilde in np.arange(0.5, 4, 0.5):
                OGD[D_tilde].append(r)
        if d['algo']['name'] == 'CP_OGD':
            CPOGD[D_tilde].append(r)
        if d['algo']['name'] == 'CS_OGD':
            CSOGD[D_tilde].append(r)
        if d['algo']['name'] == 'ADAGRAD':
            for D_tilde in np.arange(0.5, 4, 0.5):
                ADAGRAD[D_tilde].append(r)
        if d['algo']['name'] == 'CRDG':
            CRDG[D_tilde].append(r)
    except:
        print(yaml_file)
        pass

tot = {
        'OGD': OGD,
        'CP_OGD': CPOGD,
        'CS_OGD': CSOGD,
        'ADAGRAD': ADAGRAD,
        'CRDG': CRDG
        }

keys = [k for k, _ in sorted(label.items(), key=lambda x: x[1])]

plt.figure()
for k in keys:
    T = np.arange(len(L[k][1]))*d['checkpoints']
    idx = np.arange(0, 1050, 10)
    # B = (1+alpha)*np.array(list(D.values())) - L[k]
    B = [(1+alpha)*D[s] - L[k][s] for s in D.keys()]
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, B, idx=idx, speed=args.s)
    plt.plot(T, Y, label=label[k], color=colors[k], marker=marker[k],
             markevery=10, linestyle=linestyle[k], markersize=8)
    plt.fill_between(T, LB, UB, alpha=0.2, color=colors[k])
plt.legend()
plt.xlim(right=T[-1]-30)
plt.title('Fig2(b) OLR Budget')
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='k', linewidth=0.8)
plt.xlabel(r"$t$")
plt.ylabel(r"$Z_t$")
plt.tight_layout()
plt.show(block=False)

plt.figure()
for k in keys:
    T = np.arange(len(R[k][0]))*d['checkpoints']
    idx = np.arange(0, 10050, 100)
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, R[k], idx=idx, speed=args.s)
    plt.plot(T, Y, color=colors[k], label=label[k], linestyle=linestyle[k],
             marker=marker[k], markevery=10, markersize=8)
    plt.fill_between(T, LB, UB, alpha=0.2, color=colors[k])
# plt.legend()
plt.title('Fig2(a) OLR Regret')
plt.xlim(right=T[-1]-30)
plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles='dotted', color='k', linewidth=0.8)
plt.xlabel(r"$t$")
plt.ylabel(r"$R_t$")
plt.tight_layout()
plt.show(block=False)

plt.figure()
for k in keys:
    T = list(tot[k].keys())
    T.sort()
    A = []
    for s in range(len(tot[k][1])):
        tmp = []
        for d in np.arange(0.5, 4, 0.5):
            tmp.append(tot[k][d][s])
        A.append(np.array(tmp))

    tot[k] = list(tot[k].values())
    T, Y, LB, UB = utils.compute_mean_and_CI_bstr_vector(T, A, speed=1)
    if k in ['OGD', 'ADAGRAD']:
        Y = np.mean(Y)*np.ones_like(Y)
        LB = np.mean(LB)*np.ones_like(LB)
        UB = np.mean(UB)*np.ones_like(UB)
    plt.plot(T, Y, label=label[k], color=colors[k], marker=marker[k], markevery=1, linestyle=linestyle[k], markersize=8)
    plt.fill_between(T, LB, UB, alpha=0.2, color=colors[k])
plt.title('Fig3 Terminal Regret')
plt.ylim(top=10000)
plt.ylim(bottom=-100)
plt.legend()
plt.ylabel(r'$R_T$')
plt.xlabel(r'$\tilde {D}$')

plt.show()
