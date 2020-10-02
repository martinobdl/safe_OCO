from strategy import Strategy


class ADAGRAD(Strategy):

    def __init__(self, x0, alpha=0.001):
        super().__init__()
        self.x0 = x0
        self.n = len(x0)
        self.alpha = alpha
        self.x_t = x0
        self.eps = 1e-8
        self.restart()

    def _forward(self, feedback):
        self.t += 1
        g_t = feedback["grad_t"]
        self.v += np.sqrt(g_t**2)
        self.x_t = self.x_t - self.alpha * g_t/(self.v + self.eps)
        prediction = {}
        prediction["x_t"] = self.x_t
        return prediction

    def restart(self):
        self.x_t = self.x0
        self.t = 0
        self.v = np.zeros(shape=n)

    def to_dict(self):
        return {
                "name": "ADAGRAD",
                "beta_1": self.beta_1,
                "beta2": self.beta_2,
                "alpha": self.alpha,
                "eps": self.eps,
                "x0": str(self.x0)
                }


if __name__ == "__main__":

    import numpy as np
    # from linear_regression import OLR
    from IMDB_env import IMDB
    from experiment import Experiment
    import utils

    n = 10000
    x0 = np.ones(n)/n
    algo = ADAGRAD(x0)
    # env = OLR(n, max_T=1000)
    env = IMDB()
    exp = Experiment(algo, env)
    exp.run()

    print("ADAM: ", utils.accuracy(env, algo.x_t))
    print("Best: ", utils.accuracy(env, env.beta_best))
