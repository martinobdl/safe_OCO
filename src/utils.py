import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def range_to_idx(array):
    return np.sort(np.array(list(set(np.round(array)))).astype('int'))


def compute_mean_and_CI_bstr_vector(T, list_of_samples_per_seed, idx=None, alpha=0.05, speed=1):
    V, LB, UB = [], [], []
    samples = np.array(list_of_samples_per_seed).T
    if idx is not None:
        idx = range_to_idx(idx)
        T = T[idx]
        samples = samples[idx]

    for sample in samples:
        v, lb, ub = compute_mean_and_CI_bstr(sample, alpha, speed)
        V.append(v)
        LB.append(lb)
        UB.append(ub)
    return T, np.array(V), np.array(LB), np.array(UB)


def compute_mean_and_CI_bstr(samples, alpha=0.05, speed=1):
    assert speed in [1, 2, 3], "Speed must be between 1 and 3. 1 being slow and 1 very fast."
    if speed == 1:
        it = 3700
    elif speed == 2:
        it = 1000
    else:
        it = None

    if it is not None:
        bsr = bs.bootstrap(samples, stat_func=bs_stats.mean, alpha=alpha, num_iterations=it)
        return (bsr.value, bsr.lower_bound, bsr.upper_bound)
    else:
        return (np.mean(samples), 0, 0)


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
