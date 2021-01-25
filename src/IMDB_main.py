from OGD import OGD
from DPWrap import DPWRAP, CWRAP
from ADAGRAD import ADAGRAD
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

    env = SafeIMDB(times=10, rnd=rnd)

    check_point = 10
    folder = "experiments2/IMDB"

    base = OGD(x0, K_0, projection=None)
    cwrap = CWRAP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    dpwrap_ogd = DPWRAP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    adagrad = ADAGRAD(x0=x0)

    print("running DPWRAP")
    exp = Experiment(dpwrap_ogd, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)

    print("running OGD")
    exp2 = Experiment(base, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)

    print("running ADAGRAD")
    exp6 = Experiment(adagrad, env, check_point=check_point)
    exp6.run()
    exp6.save(folder=folder)

    print("accuracy DPWRAP: ", utils.accuracy(env, dpwrap_ogd.x_t))
    print("accuracy OGD: ", utils.accuracy(env, base.x_t))
    print("accuracy def: ", utils.accuracy(env, env.beta_def))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
