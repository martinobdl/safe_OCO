from env import Env
import numpy as np
import utils


class SPAM(Env):
    def __init__(self, times=1, rnd=1):
        self.rnd = rnd
        self.X = np.load('./data/spam.npy')
        self.target = np.load('./data/spam_target.npy')
        self.lam = 0
        self.beta_best = np.load('./data/beta_logistic_spam_best.npy')[0]
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
                    "grad_t": 0,
                    }
        return feedback

    def done(self):
        return self.t >= self.max_T

    def to_dict(self):
        return {
                "name": "SPAMLogistic",
                "beta_best": str(self.beta_best),
                "T": self.max_T,
                "lambda": self.lam,
                "rnd": self.rnd
                }


class SafeSPAM(SPAM):
    def __init__(self, times=1, rnd=1):
        super().__init__(times, rnd)
        self.beta_def = np.load('./data/beta_logistic_spam_weak.npy')[0]

    def step(self, prediction):
        idx = np.random.randint(0, len(self.X))
        x = self.X[idx, :]
        y = self.target[idx]
        H = prediction["x_t"]
        if "x_LR" in prediction.keys():
            H_LR = prediction["x_LR"]
        else:
            H_LR = prediction["x_t"]

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
                "name": "SPAMLogisticSafe",
                "beta_best": str(self.beta_best),
                "beta_def": str(self.beta_def),
                "T": self.max_T,
                "lambda": self.lam
                }

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


if __name__ == "__main__":
    from DPOGD import DPOGD
    from experiment import Experiment

    n = 57
    x0 = np.ones(n)/n

    alpha = 0.01
    e_l = 0+1e-4
    nk = n
    c = 2
    e_u = 10  # nk*c
    G = (nk)**0.5*1700
    D = (n*c*2)**0.5
    K_0 = D/G/2**0.5

    env = SafeSPAM(times=500)
    algo = DPOGD(x0, K_0, alpha, G, D, e_l, e_u, projection=None)
    exp = Experiment(algo, env, check_point=10000000)
    exp.run()

    print("accuracy algo: ", utils.accuracy(env, algo.x_t))
    print("accuracy algo: ", utils.accuracy(env, algo.base.x_t))
    print("accuracy best :", utils.accuracy(env, env.beta_best))
