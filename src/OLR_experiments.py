from OGD import OGD
from experiment import Experiment
from linear_regression import SafeOLR
from constant_uniform import ConstantStrategy
from strategy import SafeStartegyHybrid, SafeStrategy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import numpy as np

    n = 2
    alpha = 0.01
    x0 = np.ones(n)/n
    base = OGD(x0, 0.1)
    baseline = ConstantStrategy(x0=[0.7, 0.3])
    D = 2**0.5
    G = 2*((2*n)**0.5 + 0.1)*n**0.5
    e_l = 0.
    e_u = (2*n)**0.5 + 0.1

    algo = SafeStartegyHybrid(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    algo_naive = SafeStrategy(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    env = SafeOLR(baseline, n, max_T=100000, beta=[0.85, 0.15])

    exp = Experiment(algo, env)
    exp.run()
    exp.save()

    exp2 = Experiment(base, env)
    exp2.run()
    exp2.save()

    exp3 = Experiment(algo_naive, env)
    exp3.run()
    exp3.save()

    L = np.cumsum(exp.history['loss_t'])
    L_bar = np.cumsum(exp.history['best_loss_t'])
    L_tilde = np.cumsum(exp.history['loss_def_t'])

    L2 = np.cumsum(exp2.history['loss_t'])
    L_bar2 = np.cumsum(exp2.history['best_loss_t'])
    L_tilde2 = np.cumsum(exp2.history['loss_def_t'])

    L3 = np.cumsum(exp3.history['loss_t'])
    L_bar3 = np.cumsum(exp3.history['best_loss_t'])
    L_tilde3 = np.cumsum(exp3.history['loss_def_t'])

    T = np.arange(env.max_T)

    plt.figure()
    plt.subplot(221)
    plt.plot((L-L_bar)/T, label=r'DPOGD')
    plt.plot((L2-L_bar2)/T, label=r'OGD')
    plt.plot((L3-L_bar3)/T, label=r'COGD')
    plt.ylabel(r'$R_t/t$')
    plt.xlabel(r'$t$')
    plt.yscale('log')
    plt.legend()

    plt.subplot(222)
    plt.plot(L_tilde*(1+alpha)-L, label=r'DPOGD')
    plt.plot(L_tilde2*(1+alpha)-L2, label=r'OGD')
    plt.plot(L_tilde3*(1+alpha)-L3, label=r'COGD')
    plt.ylabel(r'$Budget$')
    plt.xlabel(r'$t$')
    plt.legend()

    plt.subplot(212)
    plt.plot(exp.history['beta'], label=r'DPOGD')
    plt.plot(0*T, label=r'OGD')
    plt.plot(exp3.history['beta'], label=r'COGD')
    plt.ylabel(r'$\beta_t$')
    plt.xlabel('t')
    plt.legend()
    plt.show()

    print("base: ", base.x_t)
    print("DPOGD: ", algo.x_t)
    print("COGD: ", algo_naive.x_t)
    print("true: ", env.beta)
