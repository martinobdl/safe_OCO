from strategy import Strategy
import utils


class OGD(Strategy):

    def __init__(self, x0, beta_0=0.5):
        super().__init__()
        self.beta_0 = beta_0
        self.x0 = x0

    def _forward(self, feedback):
        self.t += 1
        grad_t = feedback["grad_t"]
        x_t1 = utils.project(self.x_t - self.beta_0*grad_t/self.t**0.5)
        self.x_t = x_t1
        prediction = {}
        prediction["x_t"] = x_t1
        return prediction

    def restart(self):
        self.x_t = self.x0
        self.t = 0

    def to_dict(self):
        return {
                "name": "OGD",
                "beta_0": self.beta_0,
                "x0": str(self.x0)
                }


if __name__ == "__main__":

    import numpy as np
    from linear_regression import OLR
    from experiment import Experiment

    n = 2
    x0 = np.ones(n)/n
    algo = OGD(x0)
    env = OLR(2, max_T=1000)
    exp = Experiment(algo, env)
    exp.seed(1)
    exp.run()

    print("True: ", env.beta)
    print("OGD: ", algo.x_t)
