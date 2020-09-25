import numpy as np


class Env:
    def __init__(self):
        pass

    def step(self, prediction):

        # feedback is the dictionary that the algo
        # will use at time t to predict the t+1 outcome
        feedback = {
                    "loss_t": None,
                    "grad_t": None
                    }
        return feedback

    def done(self):
        return True

    def restrat(self):
        pass

    def seed(self):
        np.random.seed(self.rnd)

    def to_dict(self) -> dict:
        raise NotImplementedError
