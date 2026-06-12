import numpy as np


def pm_mutation(gp, pop_dec, *args):

    pro_m, dis_m = 1, 20
    n, d = np.shape(pop_dec)
    lower, upper = np.tile(gp.pro.lower, (n, 1)), np.tile(gp.pro.upper, (n, 1))
    site = np.random.random((n, d)) < pro_m / d
    mu = np.random.random((n, d))
    temp = site & (mu <= 0.5)
    pop_dec = np.minimum(np.maximum(pop_dec, lower), upper)
    norm = (pop_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - pop_dec[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                               1. / (dis_m + 1.)))
    off_dec = np.maximum(np.minimum(pop_dec, upper), lower)
    if len(args) != 0:
        return off_dec
    else:
        return gp.pro.fit(off_dec)
