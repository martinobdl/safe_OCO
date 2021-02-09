from OGD import OGD
from CS import CS
from CP import CP
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

    def projection(x):

        return utils.project_fixed(x, c)

    check_point = 10
    folder = "experiments/IMDB"

    base = OGD(x0, K_0, projection=projection)
    cs = CS(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cp = CP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    adagrad = ADAGRAD(x0=x0)

    print("running CP")
    exp1 = Experiment(cp, env, check_point=check_point)
    exp1.run()
    exp1.save(folder=folder)

    print("running CS")
    exp2 = Experiment(cs, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)

    print("running OGD")
    exp3 = Experiment(base, env, check_point=check_point)
    exp3.run()
    exp3.save(folder=folder)

    print("running ADAGRAD")
    exp4 = Experiment(adagrad, env, check_point=check_point)
    exp4.run()
    exp4.save(folder=folder)
