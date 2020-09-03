import numpy as np


class Experiment:
    def __init__(self, algo, env):
        self.algo = algo
        self.env = env
        self.algo.restart()
        self.env.restart()
        self.history = {}

    def seed(self, rnd):
        pass

    def run(self):
        x_t = self.algo.x_t
        while not self.env.done():
            feedback = self.env.step(x_t)
            x_t1 = self.algo(feedback)
            x_t = x_t1
            z = {**feedback, **{"x_t1": x_t1}}
            self.add_to_history(z)

    def add_to_history(self, dictionary):
        for d in dictionary.keys():
            if d not in self.history.keys():
                self.history[d] = np.array([[]])
            axis = int(self.history[d].shape == (1, 0))
            self.history[d] = np.append(self.history[d], [dictionary[d]], axis=axis)

    def save(self):
        raise NotImplementedError
