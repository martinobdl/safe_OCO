from env import Env
import numpy as np
import utils


class GBM(Env):
    def __init__(self, assets_idxs=[0, 1, 2, 3, 4]):
        self.assets_idxs = assets_idxs
        self.n_assets = len(assets_idxs)
        self.restart()
        # self.R = np.load('./data/NYSE_O.npy')[:, assets_idxs]
        self.R = np.load('./data/SP.npy')[:, assets_idxs]
        self.max_T = self.R.shape[0]
        self.min_loss = np.log(0.96)
        self.name = "Fin"

    def step(self, prediction):
        x_t = prediction["x_t"]
        r_t = self.R[self.t % self.R.shape[0], :]
        r_t = utils.clip(r_t, 0.96, 1.04)
        loss_t = np.array([-np.log(np.dot(r_t, x_t))]) - self.min_loss
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
        feedback = {
                    "loss_t": 0,
                    "grad_t": 0
                    }
        return feedback

    def done(self):
        return self.t >= self.max_T

    def to_dict(self):
        return {
                "nam": self.name,
                "assets": str(self.assets_idxs)
                }


class GBM_safe(GBM):
    def __init__(self, base_strategy=None, assets_idxs=[0, 1, 2, 3, 4]):
        self.n_assets = len(assets_idxs)
        if base_strategy is None:
            self.crp = np.ones(self.n_assets)/self.n_assets
        else:
            assert len(base_strategy) == self.n_assets
            assert abs(np.sum(base_strategy) - 1) < 1e-3
            self.crp = base_strategy
        super(GBM_safe, self).__init__(assets_idxs)
        self.name = "fin_safe"

    def step(self, prediction):
        x_t = prediction["x_t"]
        r_t = self.R[self.t % self.R.shape[0], :]
        # r_t = np.random.uniform(0.8, 1.2, size=self.n_assets)
        loss_t = np.array([-np.log(np.dot(r_t, x_t))]) - self.min_loss
        grad_t = -r_t/np.dot(r_t, x_t)
        loss_def_t = np.array([-np.log(np.dot(r_t, self.crp))]) - self.min_loss

        feedback = {
                    "loss_t": loss_t,
                    "grad_t": grad_t,
                    "recc_t": self.crp,
                    "loss_def_t": loss_def_t
                    }

        self.t += 1

        return feedback

    def restart(self):
        self.t = 0
        feedback = {
                    "loss_t": 0,
                    "grad_t": 0,
                    "recc_t": self.crp,
                    "loss_def_t": 0
                    }
        return feedback
