import numpy as np


def ens_ss(pop_obj, n_sort):
    _, b, loc = np.unique(pop_obj, return_index=True, return_inverse=True, axis=0)
    pop_obj = pop_obj[b, :]
    table = np.histogram(loc, bins=range(np.max(loc)+2))[0]
    n, m_obj = pop_obj.shape
    rank = np.arange(n)
    front_no = np.inf * np.ones(n,dtype=int)
    max_front = 0
    while np.sum(table[front_no < np.inf]) < min(n_sort, len(loc)):
        max_front += 1
        for i in range(n):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i, 0, -1):
                    if front_no[j - 1] == max_front:
                        m = 2
                        while (m <= m_obj) and (pop_obj[i, m - 1] >= pop_obj[j - 1, m - 1]):
                            m += 1
                        dominated = m > m_obj
                        if dominated or (m_obj == 2):
                            break
                if not dominated:
                    front_no[i] = max_front
        front_no[rank] = front_no
    return front_no[loc], max_front


def t_ens(pop_obj, n_sort):
    pop_obj, loc = np.unique(pop_obj, axis=0, return_inverse=True)
    table = np.histogram(loc, bins=range(np.max(loc)+2))[0]
    n, m_obj = np.shape(pop_obj)
    front_no = np.inf * np.ones(n)
    max_front = 0
    forest = np.zeros(n, dtype=int)
    children = np.zeros((n, m_obj-1), dtype=int)
    l_child = np.zeros(n, dtype=int) + m_obj - 1
    father = np.zeros(n, dtype=int)
    brother = np.zeros(n, dtype=int) + m_obj - 1
    o_rank = np.argsort(-pop_obj[:, 1:], axis=1) + 1

    while np.sum(table[front_no < np.inf]) < min(n_sort, len(loc)):
        max_front += 1
        root = np.where(front_no == np.inf)[0][0]
        forest[max_front - 1] = root
        front_no[root] = max_front
        for p in range(n):
            if front_no[p] == np.inf:
                pruning = np.zeros(n, dtype=int)
                q = forest[max_front - 1]
                while True:
                    m = 1
                    while m < m_obj and pop_obj[p, o_rank[q, m-1]] >= pop_obj[q, o_rank[q, m-1]]:
                        m += 1
                    if m == m_obj:
                        break
                    else:
                        pruning[q] = m - 1
                        if l_child[q] <= pruning[q]:
                            q = children[q, l_child[q]]
                        else:
                            while father[q] and brother[q] > pruning[father[q]]:
                                q = father[q]
                            if father[q]:
                                q = children[father[q], brother[q]]
                            else:
                                break
                if m < m_obj:
                    front_no[p] = max_front
                    q = forest[max_front - 1]
                    while children[q, pruning[q]]:
                        q = children[q, pruning[q]]
                    children[q, pruning[q]] = p
                    father[p] = q
                    if l_child[q] > pruning[q]:
                        brother[p] = l_child[q]
                        l_child[q] = pruning[q]
                    else:
                        bro = children[q, l_child[q]]
                        while brother[bro] < pruning[q]:
                            bro = children[q, brother[bro]]
                        brother[p] = brother[bro]
                        brother[bro] = pruning[q]
    return front_no[loc], max_front


def nd_sort(pop_obj, n_sort=1):
    """

    :param pop_obj: the objective vectors
    :param n_sort: sort n_sort solutions of existing vectors
    :return: the maximum front number and the front ranking
    this function is cpu only as some subfunctions are supported by cpu
    """
    n, m = np.shape(pop_obj)
    if (m < 3) or (n < 500):
        front_no, max_front = ens_ss(pop_obj, n_sort)
    else:
        front_no, max_front = t_ens(pop_obj, n_sort)
    return front_no, max_front
