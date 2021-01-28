import numpy as np


class DPWRAP:
    def __init__(self, base, alpha, G, D, e_l, e_u):
        self.base = base
        self.D = D
        self.G = G
        self.e_l = e_l
        self.e_u = e_u
        self.restart()
        self.alpha = alpha
        self.Ca = min(self.G*self.D, self.e_u - self.e_l) - self.alpha*self.e_l
        self.name = "DPWRAP_" + base.name

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


class CWRAP(DPWRAP):

    def __init__(self, base, alpha, G, D, e_l, e_u):
        super().__init__(base, alpha, G, D, e_l, e_u)
        self.name = "CWRAP_" + self.base.name

    def b(self, feedback):
        self.Loss += feedback["loss_t"]
        self.Loss_def += feedback["loss_def_t"]
        if self.Loss >= (1+self.alpha)*self.Loss_def - self.Ca:
            return np.array([1])
        else:
            return np.array([0])


if __name__ == "__main__":
    from OGD import OGD
    import utils
    from IMDB_env import SafeIMDB
    from DPOGD import DPOGD
    from experiment import Experiment

    n = 10000
    x0 = np.ones(n)/n

    alpha = 0.01
    e_l = 0
    nk = 1100
    c = 2
    e_u = nk*c
    G = nk**0.5
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    base = OGD(x0, K_0, projection=None)
    dpwrap = DPWRAP(base, alpha, G, D, e_l, e_u)
    dpogd = DPOGD(x0, K_0, alpha, G, D, e_l, e_u)
    env = SafeIMDB()
    exp = Experiment(dpwrap, env)
    exp.run()

    breakpoint()

    print("accuracy algo: ", utils.accuracy(env, dpwrap.x_t))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
