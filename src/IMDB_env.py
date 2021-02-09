from env import Env
import numpy as np
import scipy.sparse
import utils


class IMDB(Env):
    def __init__(self, times=1, rnd=1):
        self.rnd = rnd
        self.X = scipy.sparse.load_npz('./data/IMDB_BoW.npz').toarray()  # [:1000]
        self.target = np.loadtxt('./data/IMDB_target.csv', delimiter=',')  # [:1000]
        self.lam = 0
        self.beta_best = np.load('./data/beta_logistic_IMDB_best.npy')
        self.max_T = self.X.shape[0]*times

    def step(self, prediction):
        idx = np.random.randint(0, len(self.X))
        x = self.X[idx, :]
        y = self.target[idx]
        H = prediction["x_t"]

        y_hat = utils.logit(np.dot(H, x))
        y_best = utils.logit(np.dot(self.beta_best, x))

        def loss_fnc(p):
            return -y*np.log(p) - (1-y)*np.log(1-p)

        loss_t = np.array([loss_fnc(y_hat) + self.lam/2 * np.linalg.norm(H)**2 / self.max_T])
        grad_t = x*(-y*(1-y_hat)+(1-y)*y_hat)

        feedback = {
                    "best_loss_t": np.array([loss_fnc(y_best)+self.lam/2*np.linalg.norm(self.beta_best)**2/self.max_T]),
                    "loss_t": loss_t,
                    "grad_t": grad_t
                    }
        self.t += 1

        return feedback

    def restart(self):
        self.t = 0
        self.seed()
        feedback = {
                    "best_loss_t": 0,
                    "loss_t": 0,
                    "grad_t": 0
                    }
        return feedback

    def done(self):
        return self.t >= self.max_T

    def to_dict(self):
        return {
                "name": "IMDBLogistic",
                "beta_best": str(self.beta_best),
                "T": self.max_T,
                "lambda": self.lam,
                "rnd": self.rnd
                }


class SafeIMDB(IMDB):
    def __init__(self, times=1, rnd=1):
        super().__init__(times, rnd)
        self.beta_def = np.load('./data/beta_logistic_IMDB_weak.npy')[0]

    def step(self, prediction):
        idx = np.random.randint(0, len(self.X))
        x = self.X[idx, :]
        y = self.target[idx]
        H = prediction["x_t"]
        if "x_LR" in prediction.keys():
            H_LR = prediction["x_LR"]
        else:
            H_LR = H

        y_hat = utils.logit(np.dot(H, x))
        y_best = utils.logit(np.dot(self.beta_best, x))
        y_def = utils.logit(np.dot(self.beta_def, x))
        y_LR = utils.logit(np.dot(H_LR, x))

        def loss_fnc(p):
            return -y*np.log(p) - (1-y)*np.log(1-p)

        def grad_fnc(p):
            return x*(-y*(1-p)+(1-y)*p)

        loss_t = np.array([loss_fnc(y_hat) + self.lam/2 * np.linalg.norm(H)**2 / self.max_T])
        grad_t = grad_fnc(y_LR)
        best_loss_t = np.array([loss_fnc(y_best)+self.lam/2*np.linalg.norm(self.beta_best)**2/self.max_T])
        loss_def_t = np.array([loss_fnc(y_def)+self.lam/2*np.linalg.norm(self.beta_def)**2/self.max_T])

        feedback = {
                    "recc_t": self.beta_def,
                    "loss_def_t": loss_def_t,
                    "best_loss_t": best_loss_t,
                    "loss_t": loss_t,
                    "grad_t": grad_t
                    }
        self.t += 1

        return feedback

    def restart(self):
        self.t = 0
        self.seed()
        feedback = {
                    "recc_t": self.beta_def,
                    "loss_def_t": 0,
                    "best_loss_t": 0,
                    "loss_t": 0,
                    "grad_t": 0
                    }
        return feedback

    def to_dict(self):
        return {
                "name": "IMDBLogisticSafe",
                "beta_best": str(self.beta_best),
                "beta_def": str(self.beta_def),
                "T": self.max_T,
                "lambda": self.lam
                }


if __name__ == "__main__":
    pass
