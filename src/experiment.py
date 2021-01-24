import numpy as np
from string import ascii_letters
import os
import yaml
import time
from tqdm import tqdm


class Experiment:
    def __init__(self, algo, env, check_point=1):
        self.algo = algo
        self.env = env
        self.history = {}
        self.check_point = check_point
        self.restart()

    def restart(self):
        self.algo.restart()
        feedback = self.env.restart()
        return feedback

    def run(self):
        feedback = self.restart()
        prediction = self.algo(feedback)
        count = 0
        for _ in tqdm(range(self.env.max_T)):
            feedback = self.env.step(prediction)
            prediction = self.algo(feedback)
            z = {**feedback, **prediction}
            if count % self.check_point == 0:
                self.add_to_history(z)
            count += 1

    def add_to_history(self, dictionary):
        for d in dictionary.keys():
            if d in ["beta", "loss_t", "loss_def_t", "best_loss_t"]:
                if d not in self.history.keys():
                    self.history[d] = np.array([[]])
                axis = int(self.history[d].shape == (1, 0))
                self.history[d] = np.append(self.history[d], [dictionary[d]], axis=axis)

    def save(self, name=None, folder="../experiments"):

        if name is not None:
            name = name
        else:
            name = ''.join(np.random.choice(list(ascii_letters), size=10))
            name = str(int(time.time()))

        if not os.path.exists(folder):
            os.makedirs(folder)

        if os.path.isdir(folder):
            yaml_file = os.path.join(folder, name + ".yaml")
            data_file = os.path.join(folder, name + ".npz")
            with open(yaml_file, "w") as f:
                print("saving {}".format(yaml_file))
                _ = yaml.dump({
                                "checkpoints": self.check_point,
                                "algo": self.algo.to_dict(),
                                "env": self.env.to_dict()
                                }, f)
                print("saving {}".format(data_file))
                np.savez(data_file, ** self.history)
        else:
            raise Exception("{} does not exists".format(folder))
