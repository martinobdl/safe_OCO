from OGD import OGD
from COGD import COGD
from DPOGD import DPOGD
from ADAGRAD import ADAGRAD
from experiment import Experiment
from SPAM_env import SafeSPAM


if __name__ == "__main__":
    import numpy as np

    n = 57
    x0 = np.ones(n)/n

    alpha = 0.01
    e_l = 1e-4
    nk = n
    c = 2
    e_u = 1000
    G = nk**0.5*1700
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    rnd = 1

    env = SafeSPAM(times=1, rnd=rnd)

    check_point = 10
    folder = "experiments/SPAM"

    base = OGD(x0, K_0, projection=None)
    algo_dp = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    algo_c = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    adagrad = ADAGRAD(x0=x0)

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

    print("running ADAGRAD")
    exp6 = Experiment(adagrad, env, check_point=check_point)
    exp6.run()
    # exp6.save(folder=folder)
