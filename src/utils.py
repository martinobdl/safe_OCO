import numpy as np


def logit(x):
    return clip(1/(1+np.exp(-x)), 1e-4, 1-1e-4)


def accuracy(env, param):
    y_hat = logit(np.dot(env.X, param))
    return np.sum(env.target == (y_hat > 0.5))/len(env.X)


def project(v):
    mu = np.sort(v)[::-1]
    rho = np.max(np.where(mu-(np.cumsum(mu)-1)/np.arange(1, len(mu)+1) > 0))
    theta = (np.sum(mu[:rho+1])-1)/(rho+1)
    return np.maximum(v-theta, 0)


def clip(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def clean_dump_vector(a: str) -> np.ndarray:

    def _clean(x: str, sign: bool = True) -> float:
        if x == '' or x == "...":
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
        elif "-" == x[0]:
            return _clean(x[1:], sign=False)
        elif x.replace('.', '', 1).isdigit():
            return (sign*2-1)*float(x)
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
