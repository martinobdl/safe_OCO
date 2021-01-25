from OGD import OGD
from ADAGRAD import ADAGRAD
from experiment import Experiment
from SPAM_env import SafeSPAM
from DPWrap import DPWRAP
import utils


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

    env = SafeSPAM(times=100, rnd=rnd)

    check_point = 10
    folder = "experiments2/SPAM"

    base = OGD(x0, K_0, projection=None)
    # algo_dp = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    # algo_c = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    # adam = ADAM(x0=x0, alpha=1)
    adagrad = ADAGRAD(x0=x0)
    dpwrapogd = DPWRAP(base, alpha, G, D, e_l, e_u)
    # dpogdmax = DPOGDMAX(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u, projection=None)
    base = OGD(x0, K_0, projection=None)

    print("running DPwrap")
    exp = Experiment(dpwrapogd, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)

    print("running OGD")
    exp2 = Experiment(base, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)

    # print("running ADAM")
    # exp4 = Experiment(adam, env, check_point=check_point)
    # exp4.run()
    # exp4.save(folder=folder)

    # print("running DPOGDMAX")
    # exp5 = Experiment(dpogdmax, env, check_point=check_point)
    # exp5.run()
    # exp5.save(folder=folder)

    print("running ADAGRAD")
    exp6 = Experiment(adagrad, env, check_point=check_point)
    exp6.run()
    exp6.save(folder=folder)

    print("accuracy ADAGRAD: ", utils.accuracy(env, adagrad.x_t))
    print("accuracy wrap: ", utils.accuracy(env, dpwrapogd.x_t))
    print("accuracy def: ", utils.accuracy(env, env.beta_def))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
