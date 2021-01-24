from strategy import Strategy
import utils


class OGD(Strategy):

    def __init__(self, x0, beta_0=0.5, projection='simplex'):
        super().__init__()
        self.beta_0 = beta_0
        self.x0 = x0
        self.projection = projection
        self.name = 'OGD'

    def _forward(self, feedback):
        self.t += 1
        grad_t = feedback["grad_t"]
        y_t1 = self.x_t - self.beta_0*grad_t/self.t**0.5
        if self.projection == 'simplex':
            x_t1 = utils.project(y_t1)
        else:
            x_t1 = y_t1
        self.x_t = x_t1
        prediction = {}
        prediction["x_t"] = x_t1
        return prediction

    def restart(self):
        self.x_t = self.x0
        self.t = 0

    def to_dict(self):
        return {
                "name": "OGD",
                "beta_0": self.beta_0,
                "x0": str(self.x0)
                }


if __name__ == "__main__":
    pass
