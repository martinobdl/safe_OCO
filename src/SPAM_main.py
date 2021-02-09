from OGD import OGD
from ADAGRAD import ADAGRAD
from experiment import Experiment
from SPAM_env import SafeSPAM
from CS import CS
from CP import CP
import utils
from RewardDoubling import RewardDoublingNDGuess, ConversionConstraint


if __name__ == "__main__":
    import numpy as np

    n = 57
    x0 = np.ones(n)/n

    alpha = 0.01
    e_l = 1e-4
    nk = n
    c = 2
    e_u = 100
    G = nk**0.5*3
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    rnd = 1

    env = SafeSPAM(times=100, rnd=rnd)

    check_point = 10
    folder = "experiments/SPAM"

    def projection(x):
        return utils.project_fixed(x, c)

    base = OGD(x0, K_0, projection=projection)
    adagrad = ADAGRAD(x0=x0)
    cp = CP(OGD(x0, K_0, projection=projection), alpha, G, D, e_l, e_u)
    cs = CS(OGD(x0, K_0, projection=projection), alpha, G, D, e_l, e_u)
    base = OGD(x0, K_0, projection=projection)
    algo0 = RewardDoublingNDGuess(env.beta_def, e_l/2)
    reward2 = ConversionConstraint(algo0, projection)

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

    print("running CRDG")
    exp5 = Experiment(reward2, env, check_point=check_point)
    exp5.run()
    exp5.save(folder=folder)
