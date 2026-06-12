import numpy as np


def crowding_distance(pop_obj, front_no):
    infinity = 1e+14
    n, m = np.shape(pop_obj)
    if n <= 2:
        return np.full(n, infinity)
    else:
        crowd_dis = np.zeros(n)
        fronts = np.setdiff1d(np.unique(front_no), infinity)
        for f in range(len(fronts)):
            front = np.where(front_no == fronts[f])[0]
            norm = np.max(pop_obj[front], axis=0) - np.min(pop_obj[front], axis=0)
            norm[norm == 0] = np.nan
            for i in range(m):
                rank = np.argsort(pop_obj[front, i], axis=0, kind='mergesort')
                crowd_dis[front[rank[ 0]]] = infinity
                crowd_dis[front[rank[-1]]] = infinity
                for j in range(1, len(front) - 1):
                    crowd_dis[front[rank[j]]] = crowd_dis[front[rank[j]]] + (pop_obj[front[rank[j + 1]], i] -
                                                     pop_obj[front[rank[j - 1]], i]) / norm[i]
            crowd_dis[np.isnan(crowd_dis)] = 0.0
        return crowd_dis/m
