import numpy as np


def logit(x):
    return 1/(1+np.exp(-x))


def accuracy(env, param):
    y_hat = logit(np.dot(env.X, param))
    return np.sum(env.target == (y_hat > 0.5))/len(env.X)


def project(v):
    mu = np.sort(v)[::-1]
    rho = np.max(np.where(mu-(np.cumsum(mu)-1)/np.arange(1, len(mu)+1) > 0))
    theta = (np.sum(mu[:rho+1])-1)/(rho+1)
    return np.maximum(v-theta, 0)


def clean_dump_vector(a: str) -> np.ndarray:

    def _clean(x: str) -> float:
        if x == '':
            return None
        elif ' ' in x:
            x = x.replace(' ', '')
            return _clean(x)
        elif '\n' in x:
            x = x.replace('\n', '')
            return _clean(x)
        elif '[' in x:
            x = x.replace('[', '')
            return _clean(x)
        elif ']' in x:
            x = x.replace(']', '')
            return _clean(x)
        elif x.replace('.', '', 1).isdigit():
            return float(x)
        else:
            breakpoint()
            raise Exception("{} is an invalid string".format(x))

    sol = []
    for x in a.split(' '):
        clean_x = _clean(x)
        if clean_x is not None:
            sol.append(_clean(x))
    return np.array(sol)


if __name__ == "__main__":
    pass
