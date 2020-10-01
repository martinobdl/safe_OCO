from strategy import Strategy


class ADAM(Strategy):

    def __init__(self, x0, beta_1=0.9, beta_2=0.999, alpha=0.001, eps=1e-8):
        super().__init__()
        self.x0 = x0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.eps = eps
        self.x_t = x0
        self.restart()

    def _forward(self, feedback):
        self.t += 1
        g_t = feedback["grad_t"]
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (g_t*g_t)
        hat_m_t = self.m_t / (1 - self.beta_1**self.t)
        hat_v_t = self.v_t / (1 - self.beta_2**self.t)
        self.x_t = self.x_t - self.alpha/self.t**0.5 * hat_m_t / (hat_v_t ** 0.5 + self.eps)
        prediction = {}
        prediction["x_t"] = self.x_t
        return prediction

    def restart(self):
        self.x_t = self.x0
        self.m_t = 0
        self.v_t = 0
        self.t = 0

    def to_dict(self):
        return {
                "name": "ADAM",
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
    algo = ADAM(x0)
    # env = OLR(n, max_T=1000)
    env = IMDB()
    exp = Experiment(algo, env)
    exp.run()

    print("ADAM: ", utils.accuracy(env, algo.x_t))
    print("Best: ", utils.accuracy(env, env.beta_best))
