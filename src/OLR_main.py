from OGD import OGD
from ADAGRAD import ADAGRAD
from DPWrap import DPWRAP, CWRAP
from experiment import Experiment
from linear_regression import SafeOLR
from constant_uniform import ConstantStrategy
import argparse
import utils
from PFREE import RewardDoublingNDGuess, ConversionConstraint


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    n = 10
    m = 0.5
    beta = np.array([(1-m)/(n-1)]*(n-1)+[m])
    baselinex0 = beta.copy()
    alpha = 0.01
    x0 = np.ones(n)/n
    e_l = 0
    nk = 1100
    c = 2
    e_u = nk*c
    G = nk**0.5
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    print(G*D-alpha*e_l)

    check_point = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=float, default=0.2)
    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    Dtilde = args.D
    seed = args.seed
    assert Dtilde < min([2**0.5, 2**0.5*(1-min(beta)), 2**0.5*max(beta)])
    baselinex0 = beta + Dtilde/((n-1)/n)**0.5/n
    baselinex0[np.argmax(beta)] -= Dtilde/((n-1)/n)**0.5

    print(np.linalg.norm(baselinex0 - beta))

    baseline = ConstantStrategy(x0=baselinex0)

    D = 2**0.5
    G = 2*((2*n)**0.5 + 0.1)*n**0.5
    e_l = 0.
    e_u = (2*n)**0.5 + 0.1
    folder = "experiments2/OLR"
    folder = "/tmp"

    K_0 = D/G/2**0.5

    base = OGD(x0, K_0*0.3)
    adagrad = ADAGRAD(x0=x0)
    dpwrap = DPWRAP(OGD(x0, K_0*20), alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cwrap = CWRAP(OGD(x0, K_0*20), alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    algo0 = RewardDoublingNDGuess(x0, 0.03)
    reward2 = ConversionConstraint(algo0, utils.project)
    env = SafeOLR(baseline, n, max_T=100000, beta=beta, rnd=seed)

    exp1 = Experiment(dpwrap, env, check_point=check_point)
    exp1.run()
    exp1.save(folder=folder)

    exp2 = Experiment(cwrap, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)

    exp3 = Experiment(base, env, check_point=check_point)
    exp3.run()
    exp3.save(folder=folder)

    exp4 = Experiment(adagrad, env, check_point=check_point)
    exp4.run()
    exp4.save(folder=folder)

    exp5 = Experiment(reward2, env, check_point=check_point)
    exp5.run()
    exp5.save(folder=folder)

    plt.figure()
    plt.plot(exp1.history['L_t']-exp1.history['LS_t'])
    plt.plot(exp2.history['L_t']-exp2.history['LS_t'])
    plt.plot(exp3.history['L_t']-exp3.history['LS_t'])
    plt.plot(exp4.history['L_t']-exp4.history['LS_t'])
    plt.plot(exp5.history['L_t']-exp5.history['LS_t'])
    plt.legend(['dp', 'c', 'ogd', 'adagrad'])
    plt.title('RT')
    plt.show()

    plt.figure()
    plt.plot(-(exp1.history['L_t']-exp1.history['LT_t']*(1+alpha)))
    plt.plot(-(exp2.history['L_t']-exp2.history['LT_t']*(1+alpha)))
    plt.plot(-(exp3.history['L_t']-exp3.history['LT_t']*(1+alpha)))
    plt.plot(-(exp4.history['L_t']-exp4.history['LT_t']*(1+alpha)))
    plt.plot(-(exp5.history['L_t']-exp5.history['LT_t']*(1+alpha)))
    plt.legend(['dp', 'c', 'ogd', 'adagrad'])
    plt.title('ZT')
    plt.show()

    plt.figure()
    plt.plot(exp2.history['beta'])
    plt.plot(exp1.history['beta'])
    plt.legend(['c', 'dp'])
    plt.show()
