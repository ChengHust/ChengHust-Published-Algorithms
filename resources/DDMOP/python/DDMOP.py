import numpy as np
import h5py
import os
import scipy.io as sio
from Public.lhsampling import lhsampling
from Problem.DDMOP_data.DDMOP_sub import trans_func, NR, train_net, predict_net

class DDMOP1(object):
    def __init__(self, m=9, d=11, ref_num=None, evaluated=0):
        self.m = 9
        self.d = 11
        self.lower = np.array([[0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4, -1.655, -1.808, -2, -2]])
        self.upper = np.array([[1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2, 2.345, 2.192, 2, 2]])
        self.ref_num = ref_num
        self.evaluated = evaluated
        self.name = "DDMOP1"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(self.path+".mat", "r")['initial']).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T

    def init(self, n):
        decs = np.zeros((11*self.d-1, self.d))
        inverse = np.array([5, 2, 10, 6, 7, 4, 0, 1, 3, 8, 9])
        decs[:, inverse] = self.data[:, :]
        decs = decs * (self.upper - self.lower) + self.lower
        return self.fit(decs)

    def fit(self, pop_dec):
        n, d = np.shape(pop_dec)
        pop_obj = np.zeros((n, self.m))
        pop_obj[:, 0] = 1.98 + 4.9 * pop_dec[:, 0] + 6.67 * pop_dec[:, 1] + 6.98 * pop_dec[:, 2] + \
                        4.01 * pop_dec[:, 3] + 1.78 * pop_dec[:, 4] + 0.00001 * pop_dec[:, 4] + 2.73 * pop_dec[:, 1]
        pop_obj[:, 1] = 1.16 - 0.3717 * pop_dec[:, 1] * pop_dec[:, 3] - 0.00931 * pop_dec[:, 1] * pop_dec[:, 9] - \
                        0.484 * pop_dec[:, 2] * pop_dec[:, 8] + 0.01343 * pop_dec[:, 5] * pop_dec[:, 9]
        pop_obj[:, 2] = 0.261 - 0.0159 * pop_dec[:, 0] * pop_dec[:, 1] - 0.188 * pop_dec[:, 0] * pop_dec[:, 7] - \
                        0.019 * pop_dec[:, 1] * pop_dec[:, 6] + 0.0144 * pop_dec[:, 2] * pop_dec[:, 4] + \
                        0.87570 * pop_dec[:, 4] * pop_dec[:, 9] + 0.08045 * pop_dec[:, 5] * pop_dec[:, 8] + \
                        0.00139 * pop_dec[:, 7] * pop_dec[:, 10] + 0.00001575 * pop_dec[:, 9] * pop_dec[:, 10]
        pop_obj[:, 3] = 0.214 + 0.00817 * pop_dec[:, 4] - 0.131 * pop_dec[:, 0] * pop_dec[:, 7] - \
                        0.0704 * pop_dec[:, 0] * pop_dec[:, 8] + 0.03099 * pop_dec[:, 1] * pop_dec[:, 5] - \
                        0.018 * pop_dec[:, 1] * pop_dec[:, 6] + 0.0208 * pop_dec[:, 2] * pop_dec[:, 7] + \
                        0.121 * pop_dec[:, 2] * pop_dec[:, 8] - 0.00364 * pop_dec[:, 4] * pop_dec[:, 5] + \
                        0.0007715 * pop_dec[:, 4] * pop_dec[:, 9] - 0.0005354 * pop_dec[:, 5] * pop_dec[:, 9] + \
                        0.00121 * pop_dec[:, 7] * pop_dec[:, 10] + 0.00184 * pop_dec[:, 8] * pop_dec[:, 9] - \
                        0.018 * pop_dec[:, 1] ** 2
        pop_obj[:, 4] = 0.74 - 0.61 * pop_dec[:, 1] - 0.163 * pop_dec[:, 2] * pop_dec[:, 7] + \
                        0.001232 * pop_dec[:, 2] * pop_dec[:, 9] - 0.166 * pop_dec[:, 6] * pop_dec[:, 8] + \
                        0.227 * pop_dec[:, 1] ** 2
        pop_obj[:, 5] = 109.2 - 9.9 * pop_dec[:, 1] + 6.768 * pop_dec[:, 2] + 0.1792 * pop_dec[:, 9] - \
                        9.256 * pop_dec[:, 0] * pop_dec[:, 1] - 12.9 * pop_dec[:, 0] * pop_dec[:, 7] - 11 * \
                        pop_dec[:, 1] * pop_dec[:, 7] + 0.1107 * pop_dec[:, 2] * pop_dec[:, 9] + \
                        0.0207 * pop_dec[:, 4] * pop_dec[:, 9] + 6.63 * pop_dec[:, 5] * pop_dec[:, 8] - \
                        17.75 * pop_dec[:, 6] * pop_dec[:, 7] + 22 * pop_dec[:, 7] * pop_dec[:, 8] + \
                        0.32 * pop_dec[:, 8] * pop_dec[:, 9]
        pop_obj[:, 6] = 4.72 - 0.5 * pop_dec[:, 3] - 0.19 * pop_dec[:, 1] * pop_dec[:, 2] - 0.0122 * pop_dec[:, 3] * \
                        pop_dec[:, 9] + 0.009325 * pop_dec[:, 5] * pop_dec[:, 9] + 0.000191 * pop_dec[:, 10] ** 2
        pop_obj[:, 7] = 10.58 - 0.674 * pop_dec[:, 0] * pop_dec[:, 1] - 1.95 * pop_dec[:, 1] * pop_dec[:, 7] + \
                        0.02054 * pop_dec[:, 2] * pop_dec[:, 9] - 0.0198 * pop_dec[:, 3] * pop_dec[:, 9] + \
                        0.028 * pop_dec[:, 5] * pop_dec[:, 9]
        pop_obj[:, 8] = 16.45 - 0.489 * pop_dec[:, 2] * pop_dec[:, 6] - 0.843 * pop_dec[:, 4] * pop_dec[:, 5] + \
                        0.0432 * pop_dec[:, 8] * pop_dec[:, 9]
        self.evaluated += n
        return (pop_dec, pop_obj)

    def pf(self):
        return self.PF


class DDMOP2(DDMOP1):
    def __init__(self, m=3, d=5, ref_num=None, evaluated=0):
        DDMOP1.__init__(self, m, d, ref_num, evaluated)
        self.m = 3
        self.d = 5
        self.lower = np.ones((1, self.d))
        self.upper = np.ones((1, self.d)) * 3
        self.ref_num = ref_num
        self.evaluated = evaluated
        self.name = "DDMOP2"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(self.path+".mat", "r")['initial']).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T

    def init(self, n):
        decs = np.zeros((11*self.d-1, self.d))
        inverse = np.array([2, 4, 0, 1, 3])
        decs[:, inverse] = self.data[:, :]
        decs = decs * (self.upper - self.lower) + self.lower
        return self.fit(decs)


    def fit(self, pop_dec):
        n, d = np.shape(pop_dec)
        pop_obj = np.zeros((n, self.m))
        pop_obj[:, 0] = 1640.2823 + 2.3573285 * pop_dec[:, 0] + 2.3220035 * pop_dec[:, 1] + 4.5688768 * pop_dec[:, 2] + \
                        7.7213633 * pop_dec[:, 3] + 4.4559504 * pop_dec[:, 4]
        pop_obj[:, 1] = 6.5856 + 1.15 * pop_dec[:, 0] - 1.0427 * pop_dec[:, 1] + 0.9738 * pop_dec[:, 2] + \
                        0.8364 * pop_dec[:, 3] - 0.3695 * pop_dec[:, 0] * pop_dec[:, 3] + \
                        0.0861 * pop_dec[:, 0] * pop_dec[:, 4] + 0.3628 * pop_dec[:, 1] * pop_dec[:, 3] - \
                        0.1106 * pop_dec[:, 0] ** 2 - 0.3437 * pop_dec[:, 2] ** 2 + 0.1764 * pop_dec[:, 3] ** 2
        pop_obj[:, 2] = -0.0551 + 0.0181 * pop_dec[:, 0] + 0.1024 * pop_dec[:, 1] + 0.0421 * pop_dec[:, 2] - \
                        0.0073 * pop_dec[:, 0] * pop_dec[:, 1] + 0.024 * pop_dec[:, 1] * pop_dec[:, 2] - \
                        0.0118 * pop_dec[:, 1] * pop_dec[:, 3] - 0.0204 * pop_dec[:, 2] * pop_dec[:, 3] - \
                        0.008 * pop_dec[:, 2] * pop_dec[:, 4] - 0.0241 * pop_dec[:, 1] ** 2 + 0.0109 * pop_dec[:, 3] ** 2
        self.evaluated += n
        return (pop_dec, pop_obj)

class DDMOP3(DDMOP1):
    def __init__(self, m=3, d=6, ref_num=10000, evaluated=0):
        DDMOP1.__init__(self, m, d, ref_num, evaluated)
        self.m = 3
        self.d = 6
        
        Prate = 65000
        El = 400
        w0 = 2*np.pi*50
        ws = 2*np.pi*16000
        Vdc = 800
        Rb = El ** 2 / Prate
        Iref = 141

        self.lower = np.array([[2*np.pi*Vdc/(3.2*ws*Iref),
                              0.001*Rb/w0,
                              0.001/(Rb*w0),
                              0.001/(Rb*w0),
                              0.001/(Rb*w0),
                              0.001/(Rb*w0)]])
        self.upper = np.array([[2*np.pi*Vdc/(1.6*ws*Iref),
                                0.1*El ** 2/(Prate*w0),
                                0.1*El ** 2/(Prate*w0),
                                0.05*Prate/(El ** 2*w0),
                                0.05*Prate/(El ** 2*w0),
                                0.05*Prate/(El ** 2*w0)]])
        self.name = "DDMOP3"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(self.path+".mat", "r")['initial']).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T

    def init(self, n):
        decs = np.zeros((11*self.d-1, self.d))
        inverse = np.array([1, 3, 2, 4, 0, 5])
        decs[:, inverse] = self.data[:, :]
        decs = decs * (self.upper - self.lower) + self.lower
        return self.fit(decs)

    def fit(self, pop_dec):      
        Prate = 65000
        El = 400
        w0 = 2*np.pi*50
        ws = 2*np.pi*16000
        Vdc = 800
        Rb = El ** 2 / Prate
        Iref = 141
        K = 2
        
        L = pop_dec[:, :3]
        Cf = pop_dec[:, 3:4]
        C = pop_dec[:, 4:]
        Lf = 1 /(np.tile((np.arange(1, K+1)**2) * (ws ** 2), (np.shape(C)[0], 1))*C)
        pop_obj = np.zeros((len(pop_dec), self.m))
        for i in range(self.m):
            pop_obj[:, i] = 20 * \
                np.log10(np.abs(trans_func(
                    (i+1)*ws * 1j, pop_dec, self.m-1)))
        pop_obj[:, -1] = np.sum(L, axis=1) + np.sum(Lf, axis=1)
        self.evaluated += len(pop_dec)
        return (pop_dec, pop_obj)
        
        
class DDMOP4(DDMOP3):
    def __init__(self, m=3, d=6, ref_num=10000, evaluated=0):
        DDMOP3.__init__(self, m, d, ref_num, evaluated)
        Prate = 65000
        El = 400
        w0 = 2*np.pi*50
        ws = 2*np.pi*16000
        Vdc = 800
        Rb = El ** 2 / Prate
        Iref = 141
        K = 9
        self.m = K + 1
        self.d = self.m + 3
        self.lower = np.append(np.array([[2*np.pi*Vdc/(3.2*ws*Iref), 0.001*Rb/w0]]), 
                               np.ones((1, self.m+1))*0.001/(Rb*w0),
                               axis=1)
        
        self.upper = np.append(np.array([[2*np.pi*Vdc/(1.6*ws*Iref),
                                0.1*El ** 2/(Prate*w0),
                                0.1*El ** 2/(Prate*w0)]]),
                               np.ones((1, self.m)) * 0.05*Prate/(El ** 2*w0),
                               axis=1)
        self.name = "DDMOP4"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(self.path+".mat", "r")['initial']).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T

    def init(self, n):
        decs = np.zeros((11*self.d-1, self.d))
        inverse = np.array([12,5,3,13,10,8,2,6,9,4,1,7,11]) - 1
        decs[:, inverse] = self.data[:, :]
        decs = decs * (self.upper - self.lower) + self.lower
        return self.fit(decs)

    def fit(self, pop_dec):
        Prate = 65000
        El = 400
        w0 = 2*np.pi*50
        ws = 2*np.pi*16000
        Vdc = 800
        Rb = El ** 2 / Prate
        Iref = 141
        K = 9

        L = pop_dec[:, :3]
        Cf = pop_dec[:, 3:4]
        C = pop_dec[:, 4:]
        Lf = 1 / (np.tile((np.arange(1, K+1)**2) *
                  (ws ** 2), (np.shape(C)[0], 1))*C)
        pop_obj = np.zeros((len(pop_dec), self.m))
        for i in range(self.m):
            pop_obj[:, i] = 20 * \
                np.log10(np.abs(trans_func(
                    (i+1)*ws * 1j, pop_dec, self.m-1)))
        pop_obj[:, -1] = np.sum(L, axis=1) + np.sum(Lf, axis=1)
        self.evaluated += len(pop_dec)
        return (pop_dec, pop_obj)


class DDMOP5(DDMOP1):
    def __init__(self, m=3, d=11, ref_num=None, evaluated=0):
        DDMOP1.__init__(self, m, d, ref_num, evaluated)
        self.m = 3
        self.d = 11
        self.lower = np.array(
            [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0]])
        self.upper = np.array(
            [[1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.5, 0.5, 0.5, 0.5]])
        self.ref_num = ref_num
        self.evaluated = evaluated
        self.name = "DDMOP5"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(self.path+".mat", "r")['initial']).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T

    def init(self, n):
        decs = np.zeros((11*self.d-1, self.d))
        inverse = np.array([6,10,9,7,2,8,5,3,4,1,11]) - 1
        decs[:, inverse] = self.data[:, :]
        decs = decs * (self.upper - self.lower) + self.lower
        return self.fit(decs)

    def fit(self, pop_dec):
        n = len(pop_dec)
        temp = pop_dec.copy()
        temp[:, [4, 5, 6]] = np.floor((pop_dec[:, [4, 5, 6]]-0.9) / 0.0125) * 0.0125 + 0.9
        temp[:, [7, 8, 9, 10]] = np.floor((pop_dec[:, [7, 8, 9, 10]]) / 0.1) * 0.1
        pop_obj = np.zeros((n, self.m))
        for i in range(n):
            pop_obj[i] = NR(temp[i])
        self.evaluated += n
        return (pop_dec, pop_obj)


class DDMOP6(DDMOP1):
    def __init__(self, m=2, d=11, ref_num=None, evaluated=0):
        DDMOP1.__init__(self, m, d, ref_num, evaluated)
        self.m = 2

        self.ref_num = ref_num
        self.evaluated = evaluated
        self.data_no = 1
        self.name = "DDMOP6"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(
            self.path+".mat", "r")['Dataset']['Data{}'.format(self.data_no)]).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T
        self.Yield = np.log(self.data[:, 1:]) - np.log(self.data[:, 0:-1])
        self.Risk = np.cov(self.Yield)
        self.d = len(self.Risk)
        self.lower = np.zeros((1, self.d)) - 1
        self.upper = np.zeros((1, self.d)) + 1
        

    def init(self, n):
        decs = lhsampling(n, self.d) * (self.upper - self.lower) + self.lower
        return self.fit(decs)

    def fit(self, pop_dec):
        n, d = np.shape(pop_dec)
        temp = pop_dec / np.tile(np.maximum(np.sum(np.abs(pop_dec), axis=1, keepdims=True), 1), (1, d))
        pop_obj = np.zeros((n, 2))
        for i in range(n):
            pop_obj[i, 0] = np.dot(np.dot(temp[i], self.Risk), pop_dec[i].T)
            pop_obj[i, 1] = -np.sum(np.dot(temp[i], self.Yield))
            
        self.evaluated += n
        return (pop_dec, pop_obj)


class DDMOP7(DDMOP1):
    def __init__(self, m=2, d=11, ref_num=None, evaluated=0):
        DDMOP1.__init__(self, m, d, ref_num, evaluated)
        self.m = 2

        self.ref_num = ref_num
        self.evaluated = evaluated
        self.data_no = 1
        self.hidden_no = 1
        self.name = "DDMOP7"
        self.path = r"Problem\\DDMOP_data\\{}".format(self.name)
        self.data = np.array(h5py.File(
            self.path+".mat", "r")['Dataset']['Statlog_Australian']).T
        self.PF = np.array(h5py.File(self.path+".mat", "r")['PF']).T
        Mean = np.mean(self.data[:,:-1], axis=0, keepdims=True)
        Std = np.std(self.data[:, :-1], axis=0, keepdims=True)
        self.data[:, :-1] = (self.data[:,:-1] - Mean) / Std
        self.data[:, -1] = self.data[:, -1] == self.data[0, -1]
        self.train_in = self.data[:, : -1]
        self.train_out = self.data[:, -1:]
        self.d = (np.shape(self.train_in)[1]+1) * self.hidden_no + (self.hidden_no+1) * np.shape(self.train_out)[1]
        self.lower = np.zeros((1, self.d)) - 1
        self.upper = np.zeros((1, self.d)) + 1

    def init(self, n):
        decs = (np.random.random((n, self.d))-0.5) * 2 * np.random.randint(0, 2, size=(n, self.d))
        return self.fit(decs)

    def fit(self, pop_dec):
        n, d = np.shape(pop_dec)
        temp = pop_dec
        pop_obj = np.zeros((n, 2))
        for i in range(n):
            index = (np.shape(self.train_in)[1]+1)*self.hidden_no
            W1 = temp[i, :index].reshape((-1, self.hidden_no))
            W2 = temp[i, index:].reshape((self.hidden_no+1, -1))
            W1, W2 = train_net(self.train_in, self.train_out, W1, W2, 1)
            Z = predict_net(self.train_in, W1, W2)[0]
            temp[i] = np.r_[W1, W2].T
            pop_obj[i, 0] = np.mean(temp[i] != 0)
            pop_obj[i, 1] = np.mean(np.around(Z) != self.train_out)
        self.evaluated += n
        return (pop_dec, pop_obj)
