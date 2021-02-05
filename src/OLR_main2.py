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

    n = 40
    m = 1
    beta = np.zeros(n)
    baselinex0 = beta.copy()
    alpha = 0.01
    x0 = np.random.uniform(size=n)*2*m-m
    e_l = 1e-2
    c = m
    e_u = n*c*2
    D = (n*c*2)**0.5
    G = 2*D*c
    K_0 = D/G/2**0.5

    print(G*D-alpha*e_l)

    check_point = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=float, default=1.5)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-algo', type=str, default="dpwrap")
    args = parser.parse_args()

    def projection(x):
        return utils.project_fixed(x, c)

    Dtilde = args.D
    seed = args.seed
    baselinex0 = np.arange(n)/n
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
    algo0 = RewardDoublingNDGuess(baselinex0, alpha*e_l/2+100)
    reward2 = ConversionConstraint(algo0, projection)
    env = SafeOLR(baseline, n, max_T=50000, beta=beta, rnd=seed, eps_m=1)

    if args.algo == 'dpwrap':
        algo = dpwrap
    if args.algo == 'cwrpa':
        algo = cwrap
    if args.algo == 'adagrad':
        algo = adagrad
    if args.algo == 'ogd':
        algo = base
    if args.algo == 'r2':
        algo = reward2

    exp = Experiment(algo, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)

    plt.figure()
    plt.plot(exp.history['L_t']-exp.history['LS_t'])
    plt.show()
