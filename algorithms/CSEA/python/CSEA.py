import numpy as np
from ANN_model import *
from utils import *
from csea_selection import *


class CSEA(object):
    def __init__(self, decs=None, gp=None):
        self.decs = decs
        self.gp = gp
        self.k = 6
        self.g_max = 200

    def run(self):
        population = self.gp.initialization(self.gp.pro.d*11-1)
        arc = population
        f_range = [np.min(population[1], axis=0, keepdims=True),
                 np.max(population[1], axis=0, keepdims=True)]
        while self.gp.pro.evaluated <= self.gp.evaluation:
            ref_pop, _ = rsea_selection(population, self.k, f_range, 1)
            label = get_label(population[1], ref_pop[1])
            rr = np.sum(label) / len(label)
            tr = np.min([rr, 1-rr]) * 0.5
            
            train_data, test_data = data_preprocess(population[0], label)
            net = ANNmodel(self.gp.pro.d, self.gp.pro.d*2, 0.5, 8, 0.01, 200)
            net.train(train_data[0], train_data[1])
            
            test_predict = net.predict(test_data[0])
            index_good = test_data[1] == 1
            p0 = np.sum(np.abs(1 - test_predict[index_good])) / (np.sum(index_good)+0.01)
            p1 = np.sum(np.abs(0 - test_predict[~index_good])) / (np.sum(~index_good)+0.01)
            
            off_dec = surrogate_assisted_selection(self.gp, net, p0, p1, ref_pop[0], population[0], self.g_max, tr)
            if off_dec is not None:
                arc = self.gp.comb(arc, self.gp.pro.fit(off_dec))
                r = 1 - (self.gp.pro.evaluated / self.gp.evaluation)**2
                population, f_range = rsea_selection(arc, self.gp.n, f_range, r)
            else:
                population = arc.copy()
            self.gp.print_progress(arc)
        return arc
