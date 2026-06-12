import numpy as np
from dgea_selection import *
from uniformweight import uniform_point

from ibea_selection import *


class DGEA(object):

    def __init__(self, decs=None, gp=None):
        self.decs = decs
        self.gp = gp
        self.r = 10
        self.kappa = 0.05
        self.choose = 0

    def run(self):
        population = self.gp.initialization()
        offspring = self.gp.initialization()
        v, self.gp.n = uniform_point(self.gp.n, self.gp.pro.m)
        arc = population
        
        while self.gp.pro.evaluated < self.gp.evaluation:
            population = self.gp.comb(population, offspring)
            theta = (self.gp.pro.evaluated/self.gp.evaluation)**2
            population = preselection(population, v, theta, self.r)
            offspring = generation(self.gp, population, self.r)
            if self.choose == 0:
                arc = self.gp.comb(population, offspring)
            else:
                arc = ibea_selection(self.gp.comb(population, offspring), self.gp.n, self.kappa)
        return arc
