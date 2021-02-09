import numpy as np
from CP import CP


class CS(CP):

    def __init__(self, base, alpha, G, D, e_l, e_u):
        super().__init__(base, alpha, G, D, e_l, e_u)
        self.name = "CS_" + self.base.name

    def b(self, feedback):
        self.Loss += feedback["loss_t"]
        self.Loss_def += feedback["loss_def_t"]
        if self.Loss >= (1+self.alpha)*self.Loss_def - self.Ca:
            return np.array([1])
        else:
            return np.array([0])
