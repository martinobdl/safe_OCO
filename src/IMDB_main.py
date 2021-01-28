from OGD import OGD
from DPWrap import DPWRAP, CWRAP
from ADAGRAD import ADAGRAD
from experiment import Experiment
from IMDB_env import SafeIMDB
import utils
from matplotlib import pyplot as plt
# from PFREE import RewardDoublingNDGuess, ConversionConstraint


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
    projection = lambda x: utils.project_fixed(x, c)

    check_point = 10
    folder = "experiments2/IMDB"

    base = OGD(x0, K_0, projection=projection)
    cwrap = CWRAP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    dpwrap = DPWRAP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cwrap = CWRAP(base, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    adagrad = ADAGRAD(x0=x0)
    # algo0 = RewardDoublingNDGuess(x0, 0.03)
    # reward2 = ConversionConstraint(algo0, utils.project_None)

    print("running DPWRAP")
    exp1 = Experiment(dpwrap, env, check_point=check_point)
    exp1.run()
    exp1.save(folder=folder)

    print("running CWRAP")
    exp2 = Experiment(cwrap, env, check_point=check_point)
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

    # print("running R2")
    # exp5 = Experiment(reward2, env, check_point=check_point)
    # exp5.run()
    # exp5.save(folder=folder)

    print("accuracy DPWRAP: ", utils.accuracy(env, dpwrap.x_t))
    print("accuracy CWRAP: ", utils.accuracy(env, cwrap.x_t))
    print("accuracy ADAGRAD: ", utils.accuracy(env, adagrad.x_t))
    print("accuracy OGD: ", utils.accuracy(env, base.x_t))
    # print("accuracy R2: ", utils.accuracy(env, reward2.x_t))
    print("accuracy def: ", utils.accuracy(env, env.beta_def))
    print("accuracy best :", utils.accuracy(env, env.beta_best))

    plt.figure()
    plt.plot(exp1.history['L_t']-exp1.history['LS_t'])
    plt.plot(exp2.history['L_t']-exp2.history['LS_t'])
    plt.plot(exp3.history['L_t']-exp3.history['LS_t'])
    plt.plot(exp4.history['L_t']-exp4.history['LS_t'])
    # plt.plot(exp5.history['L_t']-exp5.history['LS_t'])
    plt.legend(['dp', 'c', 'ogd', 'adagrad'])
    plt.title('RT')
    plt.show()

    plt.figure()
    plt.plot(-(exp1.history['L_t']-exp1.history['LT_t']*(1+alpha)))
    plt.plot(-(exp2.history['L_t']-exp2.history['LT_t']*(1+alpha)))
    plt.plot(-(exp3.history['L_t']-exp3.history['LT_t']*(1+alpha)))
    plt.plot(-(exp4.history['L_t']-exp4.history['LT_t']*(1+alpha)))
    # plt.plot(-(exp5.history['L_t']-exp5.history['LT_t']*(1+alpha)))
    plt.legend(['dp', 'c', 'ogd', 'adagrad'])
    plt.title('ZT')
    plt.show()

    plt.figure()
    plt.plot(exp1.history['beta'])
    plt.plot(exp2.history['beta'])
    plt.legend(['dp', 'c'])
    plt.show()
