from OGD import OGD
from ADAGRAD import ADAGRAD
from experiment import Experiment
from SPAM_env import SafeSPAM
from DPWrap import DPWRAP, CWRAP
import utils
from PFREE import RewardDoublingNDGuess, ConversionConstraint


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

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
    folder = "experiments2/SPAM"
    print(K_0)
    print(G*D-alpha*e_l)
    print(min(G*D, e_u - e_l) - alpha*e_l)

    projection = lambda x: utils.project_fixed(x, c)

    base = OGD(x0, K_0, projection=projection)
    adagrad = ADAGRAD(x0=x0)
    dpwrap = DPWRAP(OGD(x0, K_0, projection=projection), alpha, G, D, e_l, e_u)
    cwrap = CWRAP(OGD(x0, K_0, projection=projection), alpha, G, D, e_l, e_u)
    base = OGD(x0, K_0, projection=projection)
    algo0 = RewardDoublingNDGuess(env.beta_def, e_l/2)
    reward2 = ConversionConstraint(algo0, projection)

    print("running DPwrap")
    exp1 = Experiment(dpwrap, env, check_point=check_point)
    exp1.run()
    exp1.save(folder=folder)

    print("running CWrap")
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

    print("running R2")
    exp5 = Experiment(reward2, env, check_point=check_point)
    exp5.run()
    exp5.save(folder=folder)

    print("accuracy ADAGRAD: ", utils.accuracy(env, adagrad.x_t))
    print("accuracy dpwrap: ", utils.accuracy(env, dpwrap.x_t))
    print("accuracy cwrap: ", utils.accuracy(env, cwrap.x_t))
    print("accuracy r2: ", utils.accuracy(env, reward2.x_t))
    print("accuracy def: ", utils.accuracy(env, env.beta_def))
    print("accuracy best :", utils.accuracy(env, env.beta_best))

    plt.figure()
    plt.plot(exp1.history['L_t']-exp1.history['LS_t'])
    plt.plot(exp2.history['L_t']-exp2.history['LS_t'])
    plt.plot(exp3.history['L_t']-exp3.history['LS_t'])
    plt.plot(exp4.history['L_t']-exp4.history['LS_t'])
    plt.plot(exp5.history['L_t']-exp5.history['LS_t'])
    plt.legend(['dp', 'c', 'ogd', 'adagrad'])
    plt.title('RT')
    plt.show()

    plt.figure()
    plt.plot(-(exp1.history['L_t']-exp1.history['LT_t']*(1+alpha)))
    plt.plot(-(exp2.history['L_t']-exp2.history['LT_t']*(1+alpha)))
    plt.plot(-(exp3.history['L_t']-exp3.history['LT_t']*(1+alpha)))
    plt.plot(-(exp4.history['L_t']-exp4.history['LT_t']*(1+alpha)))
    plt.plot(-(exp5.history['L_t']-exp5.history['LT_t']*(1+alpha)))
    plt.legend(['dp', 'c', 'ogd', 'adagrad'])
    plt.title('ZT')
    plt.show()

    plt.figure()
    plt.plot(exp2.history['beta'])
    plt.plot(exp1.history['beta'])
    plt.legend(['c', 'dp'])
    plt.show()
