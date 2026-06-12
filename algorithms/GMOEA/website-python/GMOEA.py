from Algorithm.GMOEA.ea_real import ea_real
from Algorithm.GMOEA.GAN_model import GAN
from Algorithm.GMOEA.spea2_env import *
from Public.tournament import tournament
from Variation.PM_mutation import pm_mutation


class GMOEA(object):
    def __init__(self, decs=None, gp=None):
        self.decs = decs
        self.gp = gp
        self.w_max = 1  # Smaller the better but time-consuming
        self.k = 10
        self.batch_size = 8
        self.epoch = 100
        self.n_noise = self.gp.d
        
    def norm_dec(self, pop_dec):
        input_dec = (pop_dec - self.gp.pro.lower) / \
            (self.gp.pro.upper - self.gp.pro.lower)
        return input_dec

    def run(self):
        population = self.gp.initialization()
        net = GAN(self.gp.pro.d, self.batch_size, 0.0001, self.epoch, self.n_noise)
        while self.gp.pro.evaluated <= self.gp.evaluation:
            index = environment_selection(population, self.k)[1]
            ref_dec = population[0][index]
            pool = ref_dec / np.tile(self.gp.pro.upper, (self.k, 1))

            label = np.zeros((self.gp.n, 1))
            label[index] = 1
            input_dec = self.norm_dec(population[0])
            net.train(input_dec, label, pool)
            for i in range(self.w_max):
                if np.random.random(1) < 0.5:
                    off_dec = net.generate(ref_dec / self.gp.pro.upper, self.gp.n) * self.gp.pro.upper
                else:
                    fitness = cal_fit(population[1])
                    mating = tournament(2, self.gp.n, cal_fit(population[1])[:, np.newaxis])
                    off_dec = ea_real(self.gp, population[0][mating])
                offspring = pm_mutation(self.gp, off_dec)
                population = self.gp.comb(population, offspring)
                population = environment_selection(population, self.gp.n)[0]
                self.gp.print_progress(population)
        return population
