import numpy as np


class CP:
    def __init__(self, base, alpha, G, D, e_l, e_u):
        self.base = base
        self.D = D
        self.G = G
        self.e_l = e_l
        self.e_u = e_u
        self.restart()
        self.alpha = alpha
        self.Ca = min(self.G*self.D, self.e_u - self.e_l) - self.alpha*self.e_l
        self.name = "CP_" + base.name

    def __call__(self, feedback):
        choice = self._forward(feedback)
        return choice

    def restart(self):
        self.base.restart()
        self.x_t = self.base.x0
        self.Loss = 0
        self.Loss_def = 0
        self.t = 0
        self.D_tilde = self.D
        self.beta_old = 1

    def _forward(self, feedback):
        self.t += 1
        recc = feedback["recc_t"]
        feedback['grad_t'] *= 1-self.beta_old

        base_prediction = self.base(feedback)
        x_LR = base_prediction["x_t"]
        self.D_tilde = np.linalg.norm(x_LR - recc)
        beta = self.b(feedback)
        x_t1 = beta*recc + (1-beta)*x_LR
        self.x_t = x_t1
        prediction = {}
        prediction["x_t"] = x_t1
        prediction["x_LR"] = x_LR
        prediction["beta"] = beta
        self.beta_old = beta
        return prediction

    def b(self, feedback):
        self.Loss += feedback["loss_t"]
        self.Loss_def += feedback["loss_def_t"]
        bdgt = (1+self.alpha)*self.Loss_def - self.Loss
        D = self.D_tilde
        if bdgt >= self.Ca:
            return np.array([0])
        else:
            return min(np.array([1]), max(np.array([0]),
                       1+(self.Loss - self.Loss_def*(1+self.alpha)-self.alpha*self.e_l)/(self.G*D)))

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

