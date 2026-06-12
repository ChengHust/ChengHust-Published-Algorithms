import numpy as np
from rsea_selection import *
from EAreal import ea_real


class RSEA(object):
    def __init__(self, decs=None, gp=None):
        self.decs = decs
        self.gp = gp


    def run(self):
        population = self.gp.initialization()
        f_range = 1e+14 * np.ones((2, self.gp.pro.m))
        while self.gp.pro.evaluated < self.gp.evaluation:
            mating_pool, f_range = mating_selection(
                f_range, population[1], np.ceil(self.gp.n/2)*2)
            offspring = ea_real(self.gp, population[0][mating_pool])
            population = self.gp.comb(population, offspring)
            r = 1 - (self.gp.pro.evaluated/ self.gp.evaluation)**2
            population, f_range = rsea_selection(f_range, population, self.gp.n, r)
        return population
