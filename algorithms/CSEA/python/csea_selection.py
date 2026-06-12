import numpy as np
from NDSort import nd_sort
from radar_grid import radar_grid
from scipy.spatial.distance import cdist
from EAreal import ea_real


def surrogate_assisted_selection(gp, net, p0, p1, ref_dec, pop_dec, w_max, tr):
    off_dec = ea_real(gp, np.r_[pop_dec, ref_dec], (1, 1, 15, 5))
    label = net.predict(off_dec)[:, 0]
    a, b = tr, 1-tr
    i, n = 0, len(ref_dec)
    if (p0<0.4) | ((p1 < a) & (p0<b)):
        while i < w_max:
            index = np.argsort(-label)
            dec = off_dec[index[:n]]
            off_dec = ea_real(gp, np.r_[pop_dec, ref_dec], (1, 1, 15, 5))
            label = net.predict(off_dec)[:, 0]
            i += len(off_dec)
        off_dec = off_dec[label>0.9]
    elif (p0>b) &(p1<a):
        off_dec = None
    elif p1>b:
        while i < w_max:
            index = np.argsort(label)
            dec = off_dec[index[:n]]
            off_dec = ea_real(gp, np.r_[pop_dec, ref_dec], (1, 1, 15, 5))
            label = net.predict(off_dec)[:, 0]
            i += len(off_dec)
        off_dec = off_dec[label > 0.9]
    else:
        off_dec = off_dec[np.random.randint(len(off_dec)), np.newaxis]
    return off_dec


def normal_f(f_range, pop_obj):
    front = nd_sort(pop_obj, 1)[0]
    f_range[0] = np.min(np.r_[f_range[0], pop_obj], axis=0, keepdims=True)
    f_range[1] = np.max(pop_obj[front == 1], axis=0, keepdims=True)
    return f_range


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
