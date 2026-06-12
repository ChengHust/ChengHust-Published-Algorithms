import numpy as np
from Public.crowding_distance import crowding_distance
from Public.NDSort import nd_sort


def environment_selection(population, n):

    front_no, max_front = nd_sort(population[1], n)
    next_ = front_no < max_front
    
    crowd_dis = crowding_distance(population[1], front_no)
    
    last = np.where(front_no == max_front)[0]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[:n - int(np.sum(next_))]
    next_[last[delta_n]] = True
    population = population[0][next_], population[1][next_]
    return population, front_no[next_], crowd_dis[next_]
