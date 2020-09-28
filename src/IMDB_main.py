from OGD import OGD
from COGD import COGD
from DPOGD import DPOGD
from experiment import Experiment
from IMDB_env import SafeIMDB
import utils


if __name__ == "__main__":
    import numpy as np

    n = 10000
    x0 = np.ones(n)/n

    alpha = 0.01
    e_l = 0
    nk = 1200
    c = 2
    e_u = nk*c
    G = nk**0.5
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    env = SafeIMDB(times=100)

    check_point = 10
    folder = "experiments/IMDB"

    base = OGD(x0, K_0, projection=None)
    algo = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    algo_naive = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)

    print("running DPOGD")
    exp = Experiment(algo, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)

    print("running OGD")
    exp2 = Experiment(base, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)

    print("running COGD")
    exp3 = Experiment(algo_naive, env, check_point=check_point)
    exp3.run()
    exp3.save(folder=folder)

    print("accuracy DPOGD: ", utils.accuracy(env, algo.x_t))
    print("accuracy COGD: ", utils.accuracy(env, algo_naive.x_t))
    print("accuracy OGD: ", utils.accuracy(env, base.x_t))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
