import numpy as np
from string import ascii_letters
import os
import yaml


class Experiment:
    def __init__(self, algo, env):
        self.algo = algo
        self.env = env
        self.algo.restart()
        self.env.restart()
        self.history = {}

    def seed(self):
        self.env.seed(self.rnd)

    def run(self):
        x_t = self.algo.x_t
        prediction = {}
        prediction["x_t"] = x_t
        while not self.env.done():
            feedback = self.env.step(prediction)
            prediction = self.algo(feedback)
            z = {**feedback, **prediction}
            self.add_to_history(z)

    def add_to_history(self, dictionary):
        for d in dictionary.keys():
            if d not in self.history.keys():
                self.history[d] = np.array([[]])
            axis = int(self.history[d].shape == (1, 0))
            self.history[d] = np.append(self.history[d], [dictionary[d]], axis=axis)

    def save(self, name=None, folder="../experiments"):

        if name is not None:
            name = name
        else:
            name = ''.join(np.random.choice(list(ascii_letters), size=10))

        if os.path.isdir(folder):
            yaml_file = os.path.join(folder, name + ".yaml")
            data_file = os.path.join(folder, name + ".npz")
            with open(yaml_file, "w") as f:
                _ = yaml.dump({
                                "algo": self.algo.to_dict(),
                                "env": self.env.to_dict()
                                }, f)
                np.savez(data_file, ** self.history)
        else:
            raise Exception
