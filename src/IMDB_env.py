from env import Env
import numpy as np
import scipy.sparse
import utils


class IMDB(Env):
    def __init__(self, times=1, rng=1):
        self.X = scipy.sparse.load_npz('./data/BoW.npz').toarray()  # [:1000]
        self.target = np.loadtxt('./data/target.csv', delimiter=',')  # [:1000]
        self.lam = 0
        self.beta_best = np.load('./data/beta_logistic_IMDB_best.npy')[0]
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
        self.seed(self.rng)

    def done(self):
        return self.t >= self.max_T

    def to_dict(self):
        return {
                "name": "IMDBLogistic",
                "beta_best": str(self.beta_best),
                "T": self.max_T,
                "lambda": self.lam,
                "rng": self.rng
                }


class SafeIMDB(IMDB):
    def __init__(self, times=1):
        super().__init__(times)
        self.beta_def = np.load('./data/beta_logistic_IMDB_weak.npy')[0]

    def step(self, prediction):
        idx = np.random.randint(0, len(self.X))
        x = self.X[idx, :]
        y = self.target[idx]
        H = prediction["x_t"]
        H_LR = prediction["x_LR"]

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

    def to_dict(self):
        return {
                "name": "IMDBLogisticSafe",
                "beta_best": str(self.beta_best),
                "beta_def": str(self.beta_def),
                "T": self.max_T,
                "lambda": self.lam
                }


if __name__ == "__main__":
    from OGD import OGD
    from DPOGD import DPOGD
    from COGD import COGD
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

    algo = OGD(x0, K_0, projection=None)
    algo2 = COGD(x0, K_0, alpha, G, D, e_l, e_u, projection=None)
    algo3 = DPOGD(x0, K_0, alpha, G, D, e_l, e_u, projection=None)
    env = SafeIMDB()
    exp = Experiment(algo3, env)
    exp.run()

    print("accuracy algo: ", utils.accuracy(env, algo3.x_t))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
