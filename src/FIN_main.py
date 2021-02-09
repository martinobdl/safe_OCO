from OGD import OGD
from ADAGRAD import ADAGRAD
from experiment import Experiment
from finance_env import GBM_safe
import utils
import time
from RewardDoubling import RewardDoublingNDGuess, ConversionConstraint
from CS import CS
from CP import CP


if __name__ == "__main__":
    import numpy as np

    n = 100
    beta = -np.log(0.80)
    T = 43140
    x0 = np.ones(n)/n

    check_point = 1

    D = 2**0.5
    e_l = 0.
    u = 1.04
    ll = 0.96
    G = u/ll
    e_u = float(np.log(u)-np.log(ll))
    folder = "experiments2/FIN"
    alpha = float(beta/(T*e_u))
    print(alpha)

    K_0 = D/G/2**0.5

    tmp = np.array([0]*n)
    tmp[0] = 1
    bs = tmp/sum(tmp)

    def projection(x):
        return utils.project(x)

    base = OGD(x0, K_0, projection)
    adagrad = ADAGRAD(x0=x0)
    cp = CP(OGD(x0, K_0, projection), alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cs = CS(OGD(x0, K_0, projection), alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    algo0 = RewardDoublingNDGuess(bs, alpha*0.1/2)
    crdg = ConversionConstraint(algo0, projection)

    np.random.seed(1079)
    assets = np.random.choice(467, size=n)
    env = GBM_safe(base_strategy=bs, assets_idxs=assets)

    exp = Experiment(cp, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)
    time.sleep(1)

    exp2 = Experiment(base, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)
    time.sleep(1)

    exp3 = Experiment(cs, env, check_point=check_point)
    exp3.run()
    exp3.save(folder=folder)
    time.sleep(1)

    exp4 = Experiment(crdg, env, check_point=check_point)
    exp4.run()
    exp4.save(folder=folder)
    time.sleep(1)
