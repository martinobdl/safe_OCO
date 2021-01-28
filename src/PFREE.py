from strategy import Strategy
import numpy as np


class RewardDoubling1D(Strategy):
    def __init__(self, x0, eta_1, H):
        super().__init__()
        self.x0 = x0
        self.eta_1 = eta_1
        self.H = H
        self.name = '1ddoubling'
        self.restart()

    def _forward(self, feedback):
        self.t += 1
        grad_t = feedback["grad_t"]
        self.Q += -feedback["loss_t"]
        if self.Q < self.eta*self.H:
            self.x_t = self.x_t - self.eta * grad_t
        else:
            self.eta *= 2
            self.x_t = self.x0
        prediction = {}
        prediction["x_t"] = self.x_t
        return prediction

    def restart(self):
        self.x_t = self.x0
        self.t = 0
        self.eta = self.eta_1
        self.Q = 0

    def to_dict(self):
        return {
                "name": self.name,
                "eta_1": self.eta_1,
                "H": self.H,
                "x0": str(self.x0)
                }


class RewardDoubling1DGuess(Strategy):

    def __init__(self, x0, eps):
        super().__init__()
        self.x0 = x0
        self.eps = eps
        self.epoch = 1
        self.H = 0
        self.restart()

    def restart(self):
        self.x_t = self.x0
        self.epoch = 1
        self.H = 1
        self.Hup = 0
        self.eta = self.eps
        self.algo = RewardDoubling1D(self.x0, self.eta, self.H)

    def _forward(self, feedback):
        self.Hup += feedback["grad_t"]**2
        if self.Hup > 2**(self.epoch - 1):
            self.Hup = 0
            self.epoch += 1
            self.eta /= 2
            self.H *= 2
            self.algo = RewardDoubling1D(self.x0, self.eta, self.H)
        prediction = self.algo._forward(feedback)
        self.x_t = prediction["x_t"]
        return prediction

    def to_dict(self):
        return {
                "name": self.name,
                "eps": self.eps,
                "x0": str(self.x0)
                }


class RewardDoublingNDGuess(Strategy):

    def __init__(self, x0, eps):
        super().__init__()
        self.x0 = x0
        self.n = len(x0)
        self.eps = eps
        self.reset()
        self.name = "RewardDoubligNDGuess"

    def reset(self):
        self.algos = [RewardDoubling1DGuess(self.x0[i], self.eps/self.n) for i in range(self.n)]
        self.x_t = self.x0

    def _forward(self, feedback):
        tmp = []
        for i in range(self.n):
            grad_i = feedback["grad_t"][i]
            loss_i = grad_i*self.algos[i].x_t
            f = {}
            f['grad_t'] = grad_i
            f['loss_t'] = loss_i
            p = self.algos[i]._forward(f)
            xi = p['x_t']
            tmp.append(xi)
        self.x_t = np.array(tmp)
        prediction = {}
        prediction['x_t'] = self.x_t
        return prediction


class SmoothRewardDoubling1D(Strategy):
    def __init__(self, x0, eta):
        super().__init__()
        self.eta = eta
        self.x0 = x0
        self.restart()
        self.name = "SmoothRewardDoubling1D"

    def _forward(self, feedback):
        self.t += 1
        grad_rew = feedback["grad_t"]
        self.St += -grad_rew
        self.Gt = abs(self.St)
        if self.t == 0 or self.Gt == 0:
            self.x_t = self.x0
        else:
            sign = 1 if self.St > 0 else -1
            self.x_t = self.x0 + self.eta*sign*self.B(self.Gt, self.t + 5)

        prediction = {}
        prediction['x_t'] = self.x_t
        return prediction

    def B(self, x, y):
        return np.exp(x/y**0.5)/y**(3/2)

    def restart(self):
        self.x_t = self.x0
        self.Gt = 0
        self.St = 0
        self.t = 0


class SmoothRewardDoublingND(Strategy):

    def __init__(self, x0, eps):
        super().__init__()
        self.x0 = x0
        self.n = len(x0)
        self.eps = eps
        self.reset()
        self.name = "SmoothRewardDoublingND"

    def reset(self):
        self.algos = [SmoothRewardDoubling1D(self.x0[i], self.eps/self.n) for i in range(self.n)]
        self.x_t = self.x0

    def _forward(self, feedback):
        tmp = []
        for i in range(self.n):
            grad_i = feedback["grad_t"][i]
            f = {}
            f['grad_t'] = grad_i
            p = self.algos[i]._forward(f)
            xi = p['x_t']
            tmp.append(xi)
        self.x_t = np.array(tmp)
        prediction = {}
        prediction['x_t'] = self.x_t
        return prediction


class ConversionConstraint(Strategy):
    def __init__(self, algo, projection):
        super().__init__()
        self.algo = algo
        self.projection = projection
        self.name = "ConversionConstrained_" + self.algo.name
        self.reset()

    def reset(self):
        self.algo.reset()
        self.x_t = self.projection(self.algo.x0)

    def _forward(self, feedback):
        grad = feedback["grad_t"]*np.ones(len(self.algo.x0))
        pseudo_f = {}
        proj = self.projection(self.algo.x_t)

        def p_loss(x):
            return 0.5*(np.dot(grad, x) + np.linalg.norm(grad)*np.linalg.norm(x - proj))

        def p_loss_grad(x):
            den = np.linalg.norm(x - proj)
            if abs(den) > 1e-3:
                return 0.5*(grad + np.linalg.norm(x) * (x - proj)/den)
            else:
                return 0.5*(grad)

        pseudo_f = {}
        pseudo_f["loss_t"] = p_loss(self.algo.x_t)
        pseudo_f["grad_t"] = p_loss_grad(self.algo.x_t)
        prediction = self.algo._forward(pseudo_f)
        z_t = prediction["x_t"]
        self.x_t = self.projection(z_t)
        prediction["x_t"] = self.x_t
        return prediction

    def to_dict(self):
        return {
                "name": self.name,
                }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import utils

    T = 2**18
    eta = 0.01
    x0 = np.array([0, 0])
    # x0 = 0
    grad = np.random.uniform(size=(T, 2))*2 - np.array([0.9, 1.1])  # grad of reward
    # grad = np.random.uniform(size=(T, 1))*2-0.99
    H = np.sum(grad**2)

    Q = [0]
    X = []

    feedback = {}
    # algo0 = SmoothRewardDoublingND(x0, 1)
    algo0 = RewardDoublingNDGuess(x0, 0.01)
    algo = ConversionConstraint(algo0, utils.project)
    for t in range(T):
        x = algo.x_t
        feedback['loss_t'] = -np.dot(grad[t, :], x)
        feedback['grad_t'] = -grad[t, :]

        # feedback['loss_t'] = -grad[t]*x
        # feedback['grad_t'] = -grad[t]

        p = algo._forward(feedback)
        Q.append(Q[-1] - feedback['loss_t'])
        X.append(p['x_t'])

    X = np.array(X)

    plt.figure()
    # plt.plot(X)
    plt.plot(X[:, 0])
    plt.plot(X[:, 1])
    plt.show()

    plt.figure()
    plt.plot(Q)
    plt.show()
