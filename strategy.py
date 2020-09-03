

class Strategy:
    def __init__(self):
        pass

    def _forward(self, feedback):
        raise NotImplementedError

    def __call__(self, feedback):
        choice = self._forward(feedback)
        return choice


class SafeStrategy:
    def __init__(self, base, alpha, Ca):
        self.base = base
        self.restart()
        self.alpha = alpha
        self.ca = Ca

    def __call__(self, feedback):
        choice = self.base(feedback)
        return choice

    @property
    def x_t(self):
        return self.base.x_t

    def restart(self):
        self.loss_def = 0
        self.loss = 0

    def _forward(self, feedback):
        self.loss += self.feedback['loss_t']
        self.loss_def += self.feedback['loss_def_t']
        if 'recc_t' not in feedback.key():
            raise ValueError('The env needs to pass the raccomandation (deafult strategy)')
        if self.loss >= (1+self.alpha)*self.loss_def - self.Ca:
            pred_t1 = feedback.get('recc_t')
        else:
            pred_t1 = self.base(feedback)
        return pred_t1
