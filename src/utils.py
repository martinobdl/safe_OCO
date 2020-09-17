import numpy as np


def project(v):
    mu = np.sort(v)[::-1]
    rho = np.max(np.where(mu-(np.cumsum(mu)-1)/np.arange(1, len(mu)+1) > 0))
    theta = (np.sum(mu[:rho+1])-1)/(rho+1)
    return np.maximum(v-theta, 0)


if __name__ == "__main__":
    v = np.random.randn(10)
    assert np.abs(np.sum(project(v)) - 1) < 1e-3, f"projected in {np.sum(project(v))}"
