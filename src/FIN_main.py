from OGD import OGD
from COGD import COGD
from DPOGD import DPOGD
from DPOGDMAX import DPOGDMAX
from ADAGRAD import ADAGRAD
from experiment import Experiment
from finance_env import GBM_safe
import utils
import time


if __name__ == "__main__":
    import numpy as np

    n = 100
    beta = -np.log(0.80)
    T = 43140
    x0 = np.ones(n)/n

    check_point = 1

    D = 2**0.5
    e_l = 0.
    u = 1.04
    ll = 0.96
    G = u/ll
    e_u = float(np.log(u)-np.log(ll))
    folder = "experiments/FIN"
    alpha = float(beta/(T*e_u))
    print(alpha)

    K_0 = D/G/2**0.5

    tmp = np.array([0]*n)
    tmp[0] = 1
    bs = tmp/sum(tmp)

    base = OGD(x0, K_0)
    dpogd = DPOGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    dpogdmax = DPOGDMAX(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    cogd = COGD(x0=x0, K_0=K_0, alpha=alpha, G=G, D=D, e_l=e_l, e_u=e_u)
    adagrad = ADAGRAD(x0=x0)
    assets = utils.clean_dump_vector("""[195 251 245 296 85 2 485 106 86 157 319 54 136 336 499 173
                                         408 401 440 388 473 439 306 125 327 54 192 3 383 411 65 139
                                         17 313 447 48 299 324 131 160 351 413 89 217 339 474 188
                                         80 456 179 436 395 273 393 2 235 363 402 38 264 51 277 257
                                         30 372 223 149 322 501 130 290 190 345 434 56 371 372 347 374 185
                                         23 281 468 123 389 267 415 338 176 157 214 196  20 299 384 348 196
                                         237 236 264]""").astype('int')

    env = GBM_safe(base_strategy=bs, assets_idxs=assets)

    exp = Experiment(dpogd, env, check_point=check_point)
    exp.run()
    exp.save(folder=folder)
    time.sleep(1)

    exp2 = Experiment(base, env, check_point=check_point)
    exp2.run()
    exp2.save(folder=folder)
    time.sleep(1)

    exp3 = Experiment(cogd, env, check_point=check_point)
    exp3.run()
    exp3.save(folder=folder)
    time.sleep(1)
