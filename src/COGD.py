from strategy import SafeStrategy
from OGD import OGD


class COGD(SafeStrategy):

    def __init__(self, x0, K_0, alpha, G, D, e_l, e_u, projection='simplex'):
        base = OGD(x0, K_0, projection)
        super(COGD, self).__init__(base, alpha, G, D, e_l, e_u)
        self.name = "COGD"
