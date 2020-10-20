from OGD import OGD
from COGD import COGD
from DPOGD import DPOGD
from ADAGRAD import ADAGRAD
from experiment import Experiment
from IMDB_env import SafeIMDB


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

    env = SafeIMDB(times=1, rnd=rnd)

    check_point = 10
    folder = "experiments/IMDB_test"

    base = OGD(x0, K_0, projection=None)
    algo_dp = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    algo_c = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    adagrad = ADAGRAD(x0=x0, alpha=1)

    print("running DPOGD")
    exp = Experiment(algo_dp, env, check_point=check_point)
    exp.run()
    # exp.save(folder=folder)

    print("running OGD")
    exp2 = Experiment(base, env, check_point=check_point)
    exp2.run()
    # exp2.save(folder=folder)

    print("running COGD")
    exp3 = Experiment(algo_c, env, check_point=check_point)
    exp3.run()
    # exp3.save(folder=folder)

    print("running AdaGrad")
    exp4 = Experiment(adagrad, env, check_point=check_point)
    exp4.run()
    # exp4.save(folder=folder)
