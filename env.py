

class Env:
    def __init__(self):
        pass

    def step(self, x_t):

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

    def seed(self, rnd):
        pass
