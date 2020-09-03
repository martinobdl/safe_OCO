from strategy import Strategy
import utils
import numpy as np


class OGD(Strategy):

    def __init__(self, n, beta_0):
        super().__init__()
        self.n = n
        self.beta_0 = beta_0
        self.x_t = np.ones(n)/n

    def _forward(self, feedback):
        grad_t = feedback["grad_t"]
        x_t1 = utils.project(self.x_t - self.beta_0*grad_t)
        self.x_t = x_t1
        return x_t1

    def restart(self):
        self.x_t = np.ones(self.n)/self.n


if __name__ == "__main__":
    pass
