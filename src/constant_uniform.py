from strategy import Strategy


class ConstantStrategy(Strategy):

    def __init__(self, x0):
        super().__init__()
        self.x_t = x0

    def _forward(self, feedback):
        return {'x_t': self.x_t}

    def to_dict(self):
        return {
                "name": "constant",
                "x0": str(self.x_t)
                }
