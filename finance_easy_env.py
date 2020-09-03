from env import Env
import numpy as np


class GBM(Env):
    def __init__(self, n_assets, means, std, max_T=100):
        self.n_assets = n_assets
        self.means = np.array(means)
        self.std = np.array(std)
        self.t = 0
        self.max_T = max_T
        self.min_loss = -np.log(1.2)
        self.max_loss = -np.log(0.8)

    def step(self, x_t):
        r_t = np.random.normal(size=self.n_assets)*self.std + self.means
        r_t = np.maximum(0.8, r_t)
        r_t = np.minimum(1.2, r_t)
        loss_t = np.array([-np.log(np.dot(r_t, x_t))])
        grad_t = -r_t/np.dot(r_t, x_t)
        feedback = {
                    "loss_t": loss_t,
                    "grad_t": grad_t
                    }
        self.t += 1

        return feedback

    def seed(self, rnd):
        np.random.seed(rnd)

    def restart(self):
        self.t = 0

    def done(self):
        return self.t >= self.max_T


class GBM_safe(GBM):
    def __init__(self, n_assets, means, std, max_T, safe_strategy):
        super(GBM_safe, self).__init__(n_assets, means, std, max_T)
        self.safe_strategy = safe_strategy
        self.feedback = None

    def step(self, x_t):
        r_t = np.random.normal(size=self.n_assets)*self.std + self.means
        r_t = np.maximum(0.8, r_t)
        r_t = np.minimum(1.2, r_t)
        loss_t = np.array([-np.log(np.dot(r_t, x_t))])+-self.min_loss
        grad_t = -r_t/np.dot(r_t, x_t)
        recc_t = self.safe_strategy(self.feedback) if self.feedback is not None else self.safe_strategy.x_t
        loss_def_t = np.array([-np.log(np.dot(r_t, recc_t))])+-self.min_loss

        feedback = {
                    "loss_t": loss_t,
                    "grad_t": grad_t,
                    "recc_t": recc_t,
                    "loss_def_t": loss_def_t
                    }

        self.feedback = feedback
        self.t += 1

        return feedback
