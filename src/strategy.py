import numpy as np


class Strategy:
    def __init__(self):
        pass

    def _forward(self, feedback) -> dict:
        raise NotImplementedError

    def __call__(self, feedback):
        choice = self._forward(feedback)
        return choice

    def restart(self) -> None:
        pass

    def to_dict(self) -> dict:
        # everything we need to reproduce the algo
        pass


class SafeStrategy:
    def __init__(self, base, alpha, G, D, e_l, e_u):
        self.base = base
        self.D = D
        self.G = G
        self.e_l = e_l
        self.e_u = e_u
        self.restart()
        self.alpha = alpha
        self.Ca = min(self.G*self.D, self.e_u - self.e_l) - self.alpha*self.e_l
        self.name = "SafeStrategy"

    def __call__(self, feedback):
        choice = self._forward(feedback)
        return choice

    def restart(self):
        self.base.restart()
        self.x_t = self.base.x0
        self.Loss = 0
        self.Loss_def = 0
        self.t = 0

    def _forward(self, feedback):
        self.t += 1
        recc = feedback["recc_t"]
        base_prediction = self.base(feedback)
        x_LR = base_prediction["x_t"]
        beta = self.b(feedback)
        x_t1 = beta*recc + (1-beta)*x_LR
        self.x_t = x_t1
        prediction = {}
        prediction["x_t"] = x_t1
        prediction["x_LR"] = x_LR
        prediction["beta"] = beta
        return prediction

    def b(self, feedback):
        self.Loss += feedback["loss_t"]
        self.Loss_def += feedback["loss_def_t"]
        if self.Loss >= (1+self.alpha)*self.Loss_def - self.Ca:
            return np.array([1])
        else:
            return np.array([0])

    def to_dict(self):
        return {
                "name": self.name,
                "base": self.base.to_dict(),
                "alpha": self.alpha,
                "G": self.G,
                "D": self.D,
                "e_l": self.e_l,
                "e_u": self.e_u
                }


class SafeStrategyHybrid(SafeStrategy):

    def __init__(self, base, alpha, G, D, e_l, e_u):
        super().__init__(base, alpha, G, D, e_l, e_u)
        self.name = "SafeStrategyHybrid"

    def b(self, feedback):
        self.Loss += feedback["loss_t"]
        self.Loss_def += feedback["loss_def_t"]
        return max(np.array([0]), 1+(self.Loss - self.Loss_def*(1+self.alpha)-self.alpha*self.e_l)/(self.G*self.D))
