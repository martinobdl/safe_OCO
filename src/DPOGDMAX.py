from strategy import SafeStrategyHybrid2
from OGD import OGD


class DPOGDMAX(SafeStrategyHybrid2):

    def __init__(self, x0, K_0, alpha, G, D, e_l, e_u, projection='simplex'):
        base = OGD(x0, K_0, projection)
        super(DPOGDMAX, self).__init__(base, alpha, G, D, e_l, e_u)
        self.name = "DPOGDMAX"
