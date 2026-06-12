import numpy as np
from nsga2_selection import environment_selection
from crowding_distance import crowding_distance
from NDSort import nd_sort
from tournament import tournament
from HV import HyperVolume as HV


class LSMOF(object):
    def __init__(self, decs=None, gp=None):
        self.decs = decs
        self.gp = gp
        self.w_d = 10
        self.sub_n = 30
        self.hv = HV()  # Initialize the hypervolume indicator
        
        
    def fit_fun(self, w, direct, reference):
        n, d = np.shape(w) // np.array([1, 2])
        lower, upper = self.gp.pro.lower, self.gp.pro.upper
        fitness = np.zeros((n, 1))
        collect = [np.empty((0, self.gp.pro.d)),
                np.empty((0, self.gp.pro.m))]
        for i in range(n):
            pop_dec = np.r_[np.tile(w[i:i+1, :d].T, (1, self.gp.pro.d)) * direct[:d, :] + lower,
                            upper - np.tile(w[i:i+1, d:].T, (1, self.gp.pro.d)) * direct[d:, :]]
            offspring = self.gp.pro.fit(pop_dec)
            collect = self.gp.comb(offspring, collect)
            fitness[i] = -self.hv.compute(reference, offspring[1])
        return fitness, collect
        

    def problem_reformulation(self, population):
        upper, lower = self.gp.pro.upper, self.gp.pro.lower
        reference = np.max(population[1], axis=0, keepdims=True)
        ref_pop = environment_selection(population, self.w_d)[0]
        direction = np.r_[np.sqrt(np.sum((ref_pop[0] - lower)**2, axis=1, keepdims=True)),
                        np.sqrt(np.sum((upper - ref_pop[0])**2, axis=1, keepdims=True))]
        direct = np.r_[(ref_pop[0] - lower), (upper - ref_pop[0])
                    ] / (direction + 0.001)
        w_max = 0.5 * np.sqrt(np.sum((upper - lower)**2))
        return reference, direction, direct, w_max
        
        
    def reformulated_optimization(self, gen, population):
        w_d, n, gp = self.w_d, self.sub_n, self.gp
        reference, _, direct, w_max = self.problem_reformulation(population)
        w_0 = w_max * np.random.random((self.sub_n, 2*w_d))
        fitness, pop_new = self.fit_fun(w_0, direct, reference)
        arc = pop_new.copy()
        p_cr, beta_min, beta_max = 0.2, 0.2, 0.8
        pop = [w_0, fitness]
        temp = [np.empty((0, gp.pro.d)), np.empty((0, gp.pro.m))]
        for it in range(gen):
            for i in range(n):
                x = pop[0][i]
                A = np.random.permutation(np.setdiff1d(np.arange(n), i))
                a, b, c = A[0], A[1], A[2]
                beta = np.random.uniform(beta_min, beta_max, size=(1, 2*w_d))
                y = pop[0][a] + beta*(pop[0][b] - pop[0][c])
                y = np.minimum(np.maximum(y, 0), w_max)

                z = np.zeros((1, 2*w_d))
                j_0 = np.random.randint(2*w_d)
                for j in range(2*w_d):
                    if (j == j_0) or (np.random.random() <= p_cr):
                        z[0, j] = y[0, j]
                    else:
                        z[0, j] = x[j]
                new_dec = z
                fit, pop_new = self.fit_fun(z, direct, reference)
                self.gp.print_progress(population)
                temp = gp.comb(temp, pop_new)
                index = nd_sort(temp[1], 1)[0] == 1
                temp = [temp[0][index, :], temp[1][index, :]]
                if fit < pop[1][i]:
                    pop[0][i], pop[1][i] = new_dec, fit
        arc = gp.comb(temp, arc)
        if np.shape(arc[1])[0] > np.shape(population[0])[0]:
            index = nd_sort(arc[1], 1)[0] == 1
            arc = [arc[0][index, :], arc[1][index, :]]
        return arc
        
        
    def run(self):
        population = self.gp.initialization()
        gen = int(0.05 * self.gp.evaluation // (2*self.sub_n*self.w_d))
        
        while self.gp.pro.evaluated < self.gp.evaluation:
            if self.gp.evaluated < 0.6 * self.gp.evaluation:
                arc = self.reformulated_optimization(gen, population)
                population = self.gp.comb(arc, population)
                population = environment_selection(population, self.gp.n)[0]
            else:
                front_no = nd_sort(population[1], len(population))[0]
                crowd_dis = crowding_distance(population[1], front_no)
                fitness = np.c_[front_no[:, np.newaxis], -crowd_dis[:, np.newaxis]]
                mating_pool = tournament(2, self.gp.n, fitness)
                offspring = self.gp.operator(self.gp, population[0][mating_pool])
                population = self.gp.comb(population, offspring)
                population = environment_selection(population, self.gp.n)[0]
        return population
