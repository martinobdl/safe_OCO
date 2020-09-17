from OGD import OGD
from strategy import SafeStrategy, SafeStartegyHybrid
from finance_easy_env import GBM_safe
from experiment import Experiment
from constant_uniform import ConstantStrategy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import numpy as np
    n = 20
    alpha = 0.1
    el = -np.log(1.2)
    eu = -np.log(0.8)
    Ca = eu+el - (1 + alpha)*0
    base = OGD(n, 0.1)
    algo = SafeStrategy(base, alpha, Ca)
    default_algo = ConstantStrategy(np.ones(n)/n)
    env = GBM_safe(n, [1.01] + [1]*(n-1), [0.1]*n, 10000, default_algo)
    exp = Experiment(algo, env)
    exp.seed(1)
    exp.run()

    best_algo = ConstantStrategy([1] + [0]*(n-1))
    exp3 = Experiment(best_algo, env)
    exp3.run()
    best_loss = np.cumsum(exp3.history['loss_t'])

    eta = 10
    h_algo = SafeStartegyHybrid(base, alpha, Ca, eta)
    exp2 = Experiment(h_algo, env)
    exp2.seed(1)
    exp2.run()

    plt.figure()
    plt.subplot(121)
    plt.plot(np.cumsum(exp.history['loss_t'])-best_loss, label=r'$L_t^*$')
    plt.plot(np.cumsum(exp2.history['loss_t'])-best_loss, label=r'$L_t^*(Hybrid)$')
    plt.plot(np.cumsum(exp.history['loss_def_t'])-best_loss, label=r'$\tilde L_t$')
    plt.plot((1+alpha)*np.cumsum(exp.history['loss_def_t'])-best_loss, label=r'$(1+\alpha)\tilde L_t$')
    plt.legend()

    plt.subplot(122)
    plt.plot(exp.history['played_safe'])
    plt.plot(exp2.history['beta'])
    plt.show()
