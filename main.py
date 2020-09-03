from OGD import OGD
from strategy import SafeStrategy
from finance_easy_env import GBM_safe
from experiment import Experiment
from constant_uniform_safe import ConstantSafe
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import numpy as np
    n = 2
    alpha = 0.1
    el = -np.log(1.2)
    eu = -np.log(0.8)
    Ca = eu+el - (1 + alpha)*0
    base = OGD(n, 0.5)
    algo = SafeStrategy(base, alpha, Ca)
    default_algo = ConstantSafe([0.6, 0.4])
    env = GBM_safe(n, [1.1, 1], [0.1]*n, 10000, default_algo)
    exp = Experiment(algo, env)
    exp.run()

    plt.figure()
    plt.plot(np.cumsum(exp.history['loss_t']), label=r'$L_t^*$')
    plt.plot(np.cumsum(exp.history['loss_def_t']), label=r'$\tilde L_t$')
    plt.plot((1+alpha)*np.cumsum(exp.history['loss_def_t']), label=r'$(1+\alpha)\tilde L_t$')
    plt.legend()
    plt.show()
