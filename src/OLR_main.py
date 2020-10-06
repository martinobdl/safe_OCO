from OGD import OGD
from COGD import COGD
from DPOGD import DPOGD
from DPOGDMAX import DPOGDMAX
from ADAGRAD import ADAGRAD
from experiment import Experiment
from linear_regression import SafeOLR
from constant_uniform import ConstantStrategy
import argparse


if __name__ == "__main__":
    import numpy as np

    beta = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.85])
    baselinex0 = beta.copy()
    n = 10
    alpha = 0.01
    x0 = np.ones(n)/n

    check_point = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=float, default=0.2)
    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    Dtilde = args.D
    seed = args.seed
    assert Dtilde < min([2**0.5, 2**0.5*(1-min(beta)), 2**0.5*max(beta)])
    print(Dtilde)

    baselinex0[np.argmax(beta)] -= Dtilde*(1/2)**0.5
    baselinex0[np.argmin(beta)] += Dtilde*(1/2)**0.5
    baseline = ConstantStrategy(x0=baselinex0)

    D = 2**0.5
    G = 2*((2*n)**0.5 + 0.1)*n**0.5
    e_l = 0.
    e_u = (2*n)**0.5 + 0.1
    folder = "experiments/OLR"

    K_0 = D/G/2**0.5

    base = OGD(x0, K_0)
    dpogd = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    dpogdmax = DPOGDMAX(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cogd = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    adagrad = ADAGRAD(x0=x0)
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

    exp5 = Experiment(adagrad, env, check_point=check_point)
    exp5.run()
    exp5.save(folder=folder)
