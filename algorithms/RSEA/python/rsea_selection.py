import numpy as np
from NDSort import nd_sort
from radar_grid import radar_grid
from tournament import tournament
from scipy.spatial.distance import cdist


def normal_f(f_range, pop_obj):
    front = nd_sort(pop_obj, 1)[0]
    f_range[0] = np.min(np.r_[f_range[0], pop_obj], axis=0, keepdims=True)
    f_range[1] = np.max(pop_obj[front == 1], axis=0, keepdims=True)
    return f_range


def mating_selection(f_range, pop_obj, n):
    f_range = normal_f(f_range, pop_obj)
    n0, n = len(pop_obj), int(n//2 * 2)
    pop_obj = (pop_obj - f_range[0]) / (f_range[1] - f_range[0])
    con = np.sqrt(np.sum(pop_obj ** 2, axis=1))

    site = radar_grid(pop_obj, np.ceil(np.sqrt(n0)))[0]
    crowd_g = np.unique(site, return_counts=True)[1]
    grids = tournament(2, n, crowd_g[:, np.newaxis])
    mating_pool = np.zeros(n, dtype=int)
    for i in range(n):
        current = np.where(site == grids[i])[0]
        if len(current) == 0:
            mating_pool[i] = np.random.randint(n0)
        else:
            parents = current[np.random.randint(len(current), size=2)]
            best = np.argmin(con[parents])
            mating_pool[i] = (parents[best])
    return mating_pool, f_range

def last_selection(pop_obj, choose, n, r):
    div = np.ceil(np.sqrt(n))
    n, m = np.shape(pop_obj)
    dis1 = np.sqrt(np.sum(pop_obj**2, axis=1, keepdims=True))
    dis2 = np.sqrt(1 - (1 - cdist(pop_obj, np.eye(m), metric='cosine'))**2)
    extreme = np.argmin(dis1*dis2, axis=0)
    choose = np.logical_or(choose, np.in1d(np.arange(n), extreme))

    con = np.sqrt(np.sum(pop_obj**2, axis=1))
    con = con / np.max(con)

    site, r_loc = radar_grid(pop_obj, div)
    r_dis = cdist(r_loc, r_loc)
    r_dis[np.eye(n) > 0.5] = 1e+14

    crowd_g = np.zeros(int(np.max(site) + 1))
    if len(site[choose]) == 1:
        crowd_g[site[choose]] = 1
    else:
        a = np.unique(site[choose], return_counts=True)
        crowd_g[a[0]] = a[1]

    while np.sum(choose) < n:
        remain_s = np.where(np.logical_not(choose))[0]
        remain_g = np.unique(site[remain_s])
        best_g = crowd_g[remain_g] == np.min(crowd_g[remain_g])
        current = remain_s[np.in1d(site[remain_s], remain_g[best_g])]
        fitness = r * m * con[current] - \
            np.min(r_dis[current, :][:, choose], axis=1)
        best = np.argmin(fitness)
        choose[current[best]] = True
        crowd_g[site[current[best]]] += 1
    return choose


def rsea_selection(population, n, f_range, r):
    pop_obj = population[1]
    front_no, max_front = nd_sort(pop_obj, n)
    next_ = np.where(front_no <= max_front)[0]
    size_n, m = pop_obj.shape
    if not np.any(f_range[0] == f_range[1]):
        pop_obj = (pop_obj - f_range[0]) / (f_range[1] - f_range[0])
    select = np.in1d(next_, np.where(front_no < max_front)[0])
    choose = last_selection(pop_obj[next_], select, n, r)
    selected = next_[choose]
    population = population[0][selected], population[1][selected]
    f_range = normal_f(f_range, population[1])
    return population, f_range
