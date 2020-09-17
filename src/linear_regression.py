from env import Env
import numpy as np


class OLR(Env):
    def __init__(self, n_feature, max_T=100, beta=None, std=0.0005**0.5, eps_m=0.1, rnd=1):
        self.rnd = rnd
        self.eps_m = eps_m
        self.n_feature = n_feature
        self.max_T = max_T
        self.seed()
        self.std = std
        if beta is None:
            self.beta_tmp = np.random.uniform(size=self.n_feature)
            self.beta = self.beta_tmp / np.sum(self.beta_tmp)
        else:
            assert len(beta) == self.n_feature
            assert sum(beta) == 1
            assert sum([b >= 0 for b in beta]) == self.n_feature
            self.beta = np.array(beta)

    def step(self, prediction):
        new_x = np.random.uniform(size=self.n_feature)
        new_x[0] = 1
        self.x = new_x
        beta_hat_t = prediction["x_t"]
        y_t_hat = np.dot(beta_hat_t, self.x)
        noise = max(min(np.random.normal()*self.std, self.eps_m), -self.eps_m)
        y_t = np.dot(self.beta, self.x) + noise
        loss_t = np.array([(y_t-y_t_hat)**2])
        grad_t = 2*np.array([y_t_hat - y_t])*self.x
        feedback = {
                    "best_loss_t": np.array([noise**2]),
                    "featureres_t": self.x,
                    "loss_t": loss_t,
                    "grad_t": grad_t
                    }
        self.t += 1

        return feedback

    def seed(self):
        np.random.seed(self.rnd)

    def restart(self):
        self.t = 0
        self.x = np.random.uniform(size=self.n_feature)
        self.x[0] = 1

    def done(self):
        return self.t >= self.max_T

    def to_dict(self):
        return {
                "name": "OnlineLinearRegression",
                "features": self.n_feature,
                "beta": str(self.beta),
                "T": self.max_T,
                "seed": self.rnd,
                "eps_m": self.eps_m,
                "std": self.std
                }


class SafeOLR(OLR):
    def __init__(self, safe_strategy, n_feature, max_T=100, beta=None, std=0.0005**0.5, eps_m=0.1, rnd=1):
        super().__init__(n_feature, max_T, beta, std, eps_m, rnd)
        self.safe_strategy = safe_strategy
        self.feedback = None

    def step(self, prediction):
        new_x = np.random.uniform(size=self.n_feature)
        new_x[0] = 1
        self.x = new_x
        beta_hat_t = prediction["x_t"]
        y_t_hat = np.dot(beta_hat_t, self.x)
        noise = max(min(np.random.normal()*self.std, self.eps_m), -self.eps_m)
        y_t = np.dot(self.beta, self.x) + noise
        loss_t = np.array([(y_t-y_t_hat)**2])
        grad_t = 2*np.array([y_t_hat - y_t])*self.x

        recc_t = self.safe_strategy(self.feedback)['x_t'] if self.feedback is not None else self.safe_strategy.x_t
        loss_def_t = np.array([(y_t-np.dot(recc_t, self.x))**2])
        feedback = {
                    "best_loss_t": np.array([noise**2]),
                    "recc_t": recc_t,
                    "featureres_t": self.x,
                    "loss_t": loss_t,
                    "grad_t": grad_t,
                    "loss_def_t": loss_def_t
                    }
        self.t += 1

        return feedback

    def to_dict(self):
        return {
                "name": "OnlineLinearRegressionSafe",
                "safe_strategy": self.safe_strategy.to_dict(),
                "features": self.n_feature,
                "beta": str(self.beta),
                "T": self.max_T,
                "seed": self.rnd,
                "eps_m": self.eps_m,
                "std": self.std
                }


if __name__ == "__main__":
    pass
