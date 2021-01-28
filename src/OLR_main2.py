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

    n = 20
    m = 1
    beta = np.random.uniform(size=n)*2*m-m
    # beta[0] = m
    # beta[-1] = -m
    baselinex0 = beta.copy()
    alpha = 0.01
    x0 = np.ones(n)/n
    e_l = 1e-2
    nk = 100
    c = m
    e_u = 20
    G = nk**0.5
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    print(G*D-alpha*e_l)

    check_point = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=float, default=0.2)
    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    projection = lambda x: utils.project_fixed(x, c)

    Dtilde = args.D
    seed = args.seed
    baselinex0 = np.random.uniform(size=n)*2*m-m
    conv = 1-Dtilde/np.linalg.norm(baselinex0 - beta)
    baselinex0 = beta*conv + (1-conv)*baselinex0
    print(np.linalg.norm(baselinex0 - beta))
    assert np.linalg.norm(baselinex0 - projection(baselinex0)) < 1e-3

    baseline = ConstantStrategy(x0=baselinex0)

    folder = "experiments2/OLR"
    folder = "/tmp"

    base = OGD(x0, K_0, projection)
    adagrad = ADAGRAD(x0=x0)
    dpwrap = DPWRAP(OGD(x0, K_0, projection), alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cwrap = CWRAP(OGD(x0, K_0, projection), alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    algo0 = RewardDoublingNDGuess(baselinex0, alpha*e_l/2)
    reward2 = ConversionConstraint(algo0, projection)
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
    plt.legend(['dp', 'c', 'ogd', 'adagrad', 'r2'])
    plt.title('RT')
    plt.show()

    plt.figure()
    plt.plot(-(exp1.history['L_t']-exp1.history['LT_t']*(1+alpha)))
    plt.plot(-(exp2.history['L_t']-exp2.history['LT_t']*(1+alpha)))
    plt.plot(-(exp3.history['L_t']-exp3.history['LT_t']*(1+alpha)))
    plt.plot(-(exp4.history['L_t']-exp4.history['LT_t']*(1+alpha)))
    plt.plot(-(exp5.history['L_t']-exp5.history['LT_t']*(1+alpha)))
    plt.legend(['dp', 'c', 'ogd', 'adagrad', 'r2'])
    plt.title('ZT')
    plt.show()

    plt.figure()
    plt.plot(exp2.history['beta'])
    plt.plot(exp1.history['beta'])
    plt.legend(['c', 'dp'])
    plt.show()
