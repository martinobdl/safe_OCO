from strategy import Strategy
import numpy as np


class ConstantSafe(Strategy):

    def __init__(self, n):
        super().__init__()
        if type(n) == int:
            self.x_t = np.ones(n)/n
        else:
            self.x_t = np.array(n)

    def _forward(self, feedback):
        return self.x_t
