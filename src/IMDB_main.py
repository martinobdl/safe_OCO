from OGD import OGD
from COGD import COGD
from DPOGD import DPOGD
from DPOGDMAX import DPOGDMAX
from ADAM import ADAM
from experiment import Experiment
from IMDB_env import SafeIMDB
import utils


if __name__ == "__main__":
    import numpy as np

    n = 10000
    x0 = np.ones(n)/n

    alpha = 0.01
    e_l = 1e-4
    nk = 1200
    c = 2
    e_u = 10
    G = nk**0.5
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    rnd = 1

    env = SafeIMDB(times=4, rnd=rnd)

    check_point = 10
    folder = "experiments/IMDB_test"

    base = OGD(x0, K_0, projection=None)
    algo_dp = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    algo_c = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    adam = ADAM(x0=x0, alpha=1)
    dpogdmax = DPOGDMAX(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)

    # print("running DPOGD")
    # exp = Experiment(algo_dp, env, check_point=check_point)
    # exp.run()
    # exp.save(folder=folder)

    # print("running OGD")
    # exp2 = Experiment(base, env, check_point=check_point)
    # exp2.run()
    # exp2.save(folder=folder)

    # print("running COGD")
    # exp3 = Experiment(algo_c, env, check_point=check_point)
    # exp3.run()
    # exp3.save(folder=folder)

    print("running ADAM")
    exp4 = Experiment(adam, env, check_point=check_point)
    exp4.run()
    exp4.save(folder=folder)

    print("running ADAM")
    exp5 = Experiment(dpogdmax, env, check_point=check_point)
    exp5.run()
    exp5.save(folder=folder)

    print("accuracy DPOGD: ", utils.accuracy(env, algo_dp.x_t))
    print("accuracy DPOGDMAX: ", utils.accuracy(env, dpogdmax.x_t))
    print("accuracy COGD: ", utils.accuracy(env, algo_c.x_t))
    print("accuracy DPOGD base: ", utils.accuracy(env, algo_dp.base.x_t))
    print("accuracy COGD base: ", utils.accuracy(env, algo_c.base.x_t))
    print("accuracy OGD: ", utils.accuracy(env, base.x_t))
    print("accuracy ADAM: ", utils.accuracy(env, adam.x_t))
    print("accuracy def: ", utils.accuracy(env, env.beta_def))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
