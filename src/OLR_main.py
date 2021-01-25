from OGD import OGD
from ADAGRAD import ADAGRAD
from DPWrap import DPWRAP, CWRAP
from experiment import Experiment
from linear_regression import SafeOLR
from constant_uniform import ConstantStrategy
import argparse


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    n = 10
    beta = np.array([0.1/(n-1)]*(n-1)+[0.9])
    baselinex0 = beta.copy()
    alpha = 0.05
    x0 = np.ones(n)/n
    e_l = 0
    nk = 1100
    c = 2
    e_u = nk*c
    G = nk**0.5
    D = 2**0.5  # (n*c*2)**0.5
    K_0 = D/G/2**0.5

    print(G*D-alpha*e_l)

    check_point = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=float, default=0.2)
    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    Dtilde = args.D
    seed = args.seed
    assert Dtilde < min([2**0.5, 2**0.5*(1-min(beta)), 2**0.5*max(beta)])
    print(Dtilde)

    baselinex0 = beta + Dtilde/((n-1)/n)**0.5/n
    baselinex0[np.argmax(beta)] -= Dtilde/((n-1)/n)**0.5
    baseline = ConstantStrategy(x0=baselinex0)

    breakpoint()

    D = 2**0.5
    G = 2*((2*n)**0.5 + 0.1)*n**0.5
    e_l = 0.
    e_u = (2*n)**0.5 + 0.1
    folder = "experiments2/OLR"

    K_0 = D/G/2**0.5

    base = OGD(x0, K_0)
    # dpogd = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    # dpogdmax = DPOGDMAX(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    # cogd = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    adagrad = ADAGRAD(x0=x0)
    dpwrap = DPWRAP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    env = SafeOLR(baseline, n, max_T=100000, beta=beta, rnd=seed)

    # exp = Experiment(dpogd, env, check_point=check_point)
    # exp.run()
    # exp.save(folder=folder)

    # exp2 = Experiment(base, env, check_point=check_point)
    # exp2.run()
    # exp2.save(folder=folder)

    # exp3 = Experiment(cogd, env, check_point=check_point)
    # exp3.run()
    # exp3.save(folder=folder)

    # exp4 = Experiment(dpogdmax, env, check_point=check_point)
    # exp4.run()
    # exp4.save(folder=folder)

    # exp5 = Experiment(adagrad, env, check_point=check_point)
    # exp5.run()
    # exp5.save(folder=folder)

    exp = Experiment(dpwrap, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)

    plt.figure()
    plt.plot(exp.history['L_t']-exp.history['LS_t'])
    plt.plot(-(exp.history['L_t']-exp.history['LT_t']*(1+alpha)))
    plt.plot(exp.history['L_t']-exp.history['LT_t'])
    plt.legend(['regret', 'bdgt', 'reg_def'])
    plt.show()

    plt.figure()
    plt.plot(exp.history['beta'])
    plt.show()
