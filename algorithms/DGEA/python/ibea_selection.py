import numpy as np


def ibea_selection(population, n, kappa):
    N = np.shape(population[0])[0]
    next_ = np.arange(N, dtype=int)
    fitness, I, C = cal_fitness(population[1], kappa)

    while len(next_) > n:
        x = np.argmin(fitness[:, next_])
        fitness += np.exp(-I[next_[x]] /
                            np.tile(C[:, next_[x]], (1, N))/kappa)
        next_ = np.delete(next_, x)
    return population[0][next_, :], population[1][next_, :]


def cal_fitness(pop_obj, kappa):
    n = pop_obj.shape[0]
    pop_min = pop_obj.min(axis=0, keepdims=True)
    pop_max = pop_obj.max(axis=0, keepdims=True)
    pop_obj = (pop_obj - pop_min) / (pop_max - pop_min)
    I = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            I[i, j] = np.max(pop_obj[i] - pop_obj[j])
    C = np.max(np.abs(I), axis=0, keepdims=True)
    fitness = np.sum(-np.exp(-I/np.tile(C, (n, 1)) / kappa),
                     axis=0, keepdims=True) + 1
    return fitness, I, C
