import numpy as np


def ea_real(gp, pop_dec, *args):
    if len(args) < 4:
        pro_c, pro_m, dis_c, dis_m = 1, 1, 20, 20
    elif len(args)==4:
        pro_c, pro_m, dis_c, dis_m = args[:4]
    pop_dec = pop_dec[:(len(pop_dec)//2)*2]
    n, d = np.shape(pop_dec)
    off_n = n//2
    parent_1_dec = pop_dec[:off_n, :]
    parent_2_dec = pop_dec[off_n:, :]
    beta = np.zeros((off_n, d))
    mu = np.random.random((off_n, d))
    beta[mu <= 0.5] = np.power(2*mu[mu <= 0.5], 1/(dis_c + 1))
    beta[mu > 0.5] = np.power(2*mu[mu > 0.5], -1/(dis_c + 1))
    beta = beta * ((-1)**np.random.randint(2, size=(off_n, d)))
    beta[np.random.random((off_n, d)) < 0.5] = 1
    beta[np.tile(np.random.random((off_n, 1)) > pro_c, (1, d))] = 1
    offspring_dec = np.r_[(parent_1_dec + parent_2_dec) / 2 + beta * (parent_1_dec - parent_2_dec) / 2,
                          (parent_1_dec + parent_2_dec) / 2 - beta * (parent_1_dec - parent_2_dec) / 2]

    lower, upper = np.tile(gp.pro.lower, (off_n*2, 1)), np.tile(gp.pro.upper, (off_n*2, 1))
    site = np.random.random((off_n*2, d)) < pro_m / d
    mu = np.random.random((off_n*2, d))
    temp = site & (mu <= 0.5)
    offspring_dec = np.minimum(np.maximum(offspring_dec, lower), upper)
    norm = (offspring_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1 - norm, dis_m + 1),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring_dec[temp]) / (upper[temp] - lower[temp])
    offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                           (1 - np.power(2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1),
                                         1 / (dis_m + 1.)))
    if len(args) != 0:
        return offspring_dec
    else:
        return gp.pro.fit(offspring_dec)

