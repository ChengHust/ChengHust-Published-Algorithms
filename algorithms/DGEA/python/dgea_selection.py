import numpy as np
from NDSort import nd_sort
from scipy.spatial.distance import cdist
from PM_mutation import pm_mutation


def preselection(population, v, theta, r):
    nv = np.shape(v)[0]

    front, max_front = nd_sort(population[1],
                               np.min([nv, population[1].shape[0]]))
    if np.sum(front == 1) >= r:
        next_0 = front < max_front
        last = np.where(front == max_front)[0]
    else:
        next_0 = front == 1
        last = np.where(np.logical_not(next_0))[0]
    pop_obj = population[1][last]
    n, m = pop_obj.shape
    pop_obj = pop_obj - np.min(population[1], axis=0, keepdims=True)

    cosine = 1 - cdist(v, v, metric='cosine')
    cosine[np.eye(len(cosine)) == 1] = 0
    gamma = np.amin(np.arccos(cosine), axis=1, keepdims=True)

    angle = np.arccos(1 - cdist(pop_obj, v, metric='cosine'))
    associate = np.argmin(angle, axis=1)
    next_ = np.zeros(nv, dtype=int)

    for _, index in enumerate(np.unique(associate, return_inverse=False)):
        current = np.where(associate == index)[0]
        apd = (1 + m*theta*angle[current, index: index+1]/gamma[index]) * \
            np.sqrt(np.sum(pop_obj[current]**2, axis=1, keepdims=True))
        best = np.argmin(apd)
        next_[index] = current[best]

    choose = next_[next_ != 0]
    next_0[last[choose]] = True
    arc = population[0][next_0], population[1][next_0]
    return arc


def generation(gp, population, r):
    n, d = population[0].shape
    front_no, _ = nd_sort(population[1], n)
    index_d = np.where(front_no > 1)[0]
    domi_n = index_d.size
    index_n = np.where(front_no == 1)[0]
    non_dec = population[0][index_n]
    domi_dec = population[0][index_d]
    sub_n = np.max([gp.n//r, 10])
    pop_dec = np.empty((sub_n*r, d))
    index = np.random.permutation(n - domi_n)[0]
    start_p = non_dec[index]
    if domi_n < r:
        if n < r:
            index = np.random.permutation(n - domi_n)
        else:
            index = np.random.permutation(n - domi_n)[:r-domi_n]
        end_p = np.r_[domi_dec, non_dec[index]]
    else:
        index = np.random.permutation(domi_n)[:r]
        end_p = domi_dec[index]

    r = end_p.shape[0]
    vector = end_p - start_p + np.ones((1, d))*0.001
    direct = vector / np.sqrt(np.sum(vector**2, axis=1, keepdims=True))
    for i in range(r):
        lambda_ = np.dot(non_dec - start_p, direct[i].T)
        sigma_ = np.std(lambda_)*(1. + (domi_n+1)/n)
        off_dec = np.tile(np.random.normal(0, sigma_, (sub_n, 1)), (1, d)) * \
            np.tile(direct[i], (sub_n, 1)) + start_p
        off_dec = np.maximum(np.minimum(
            off_dec, gp.pro.upper), gp.pro.lower)
        pop_dec[i*sub_n:(i+1)*sub_n, :] = off_dec

    offspring = pm_mutation(gp, pop_dec)
    return offspring
