from strategy import SafeStrategyHybrid
from OGD import OGD


class DPOGD(SafeStrategyHybrid):

    def __init__(self, x0, K_0, alpha, G, D, e_l, e_u, projection='simplex'):
        base = OGD(x0, K_0, projection)
        super(DPOGD, self).__init__(base, alpha, G, D, e_l, e_u)
        self.name = "DPOGD"
