import numpy as np


def trans_func(s, X, n):
    L = X[:, :3]
    Cf = X[:, 3:4]
    C = X[:, 4:]
    ws = 2 * np.pi * 16000
    Lf = 1. / (np.tile((np.arange(1, n+1) ** 2)
                       * (ws ** 2), (len(C), 1)) * C)
    RL = 0.005
    Glc = np.sum(np.tile((L[:, 1:2] * s * (L[:, 2:3] * Cf * (s**2) + 1) +
                          L[:, 2:3] * s), (1, n)) / (Lf*s + RL + 1./(C*s)), axis=1)
    G = 1. / (L[:, 0] * s * Glc + L[:, 1] * s * (L[:, 2] * Cf[:, 0] * (s**2) + 1) + L[:, 2] * s)
    return G


def NR(pop_dec):
    Data1 = np.array([
            [1,  3,  1.060,   0,         0,          0,        232.3859,     -16.8888,   1.06,   0],
            [2,  2,  1.045,  -4.980,     21.700,     12.7,     40,           42.39645,   1.045,  0],
            [3,  2,  1.010,  -12.710,    94.200,     19,       0,            23.39357,   1.01,   0],
            [4,  1,  1.018,  -10.320,    47.800,     - 3.9,     0,            0,          1,      0],
            [5,  1,  1.020,  -8.7820,    7.600,      1.6,      0,            0,          1,      0],
            [6,  2,  1.070,  -14.220,    11.200,     7.5,      0,            12.24036,   1.07,   0],
            [7,  1,  1.062,  -13.360,    0,          0,        0,            0,          1,      0],
            [8,  2,  1.090,  -13.360,    0,          0,        0,            17.3566,    1.09,   0],
            [9,  1,  1.056,  -14.940,    29.500,     16.6,     0,            0,          1,      0.19],
            [10, 1,  1.051,  -15.100,    9,          5.8,      0,            0,          1,      0],
            [11, 1,  1.057,  -14.790,    3.500,      1.8,      0,            0,          1,      0],
            [12, 1,  1.055,  -15.070,    6.100,      1.6,      0,            0,          1,      0],
            [13, 1,  1.050,  -15.150,    13.500,     5.8,      0,            0,          1,      0],
            [14, 1,  1.035,  -16.030,    14.900,     5,        0,            0,          1,      0]])

    Data2 = np.array([
            [5,  6,  2, 0,         0.25202, 0,      0.932],
            [4,  7,  2, 0,         0.20912, 0,      0.978],
            [4,  9,  2, 0,         0.55618, 0,      0.969],
            [1,  2,  1, 0.01938,   0.05917, 0.0528, 0],
            [2,  3,  1, 0.04699,   0.19797, 0.0438, 0],
            [2,  4,  1, 0.05811,   0.17632, 0.0374, 0],
            [1,  5,  1, 0.05403,   0.22304, 0.0492, 0],
            [2,  5,  1, 0.05695,   0.17388, 0.034,  0],
            [3,  4,  1, 0.06701,   0.17103, 0.0346, 0],
            [4,  5,  1, 0.01335,   0.04211, 0.0128, 0],
            [7,  8,  1, 0,         0.17615, 0,      0],
            [7,  9,  1, 0,         0.11001, 0,      0],
            [9,  10, 1, 0.03181,   0.0845,  0,      0],
            [6,  11, 1, 0.09498,   0.1989,  0,      0],
            [6,  12, 1, 0.12291,   0.25581, 0,      0],
            [6,  13, 1, 0.06615,   0.13027, 0,      0],
            [9,  14, 1, 0.12711,   0.27038, 0,      0],
            [10, 11, 1, 0.08205,   0.19207, 0,      0],
            [12, 13, 1, 0.22092,   0.19988, 0,      0],
            [13, 14, 1, 0.17093,   0.34802, 0,      0]])
 
 
    Data1[[1, 2, 5, 7], 8] = pop_dec[:4]
    Data2[[0, 1, 2], 6] = pop_dec[4:7]
    Data1[[8, 9, 12, 13], 9] = pop_dec[7:11]
    baseMVA     = 100
    Vtype       = Data1[:, 1]
    Pload       = Data1[:, 4]
    Qload       = Data1[:, 5]
    Pgen        = Data1[:, 6]
    Qgen        = Data1[:, 7]
    Vset        = Data1[:, 8]
    Qsh         = Data1[:, 9]
    II          = (Data2[:, 0] - 1).astype(int)
    JJ          = (Data2[:, 1] - 1).astype(int)
    Ltype       = Data2[:, 2]
    R           = Data2[:, 3]
    X           = Data2[:, 4]
    B           = Data2[:, 5] / 2.
    K           = Data2[:, 6]

    y1 = np.zeros((14, 14)).astype(complex)
    y2 = np.zeros((14, 14)).astype(complex)
    y3 = np.zeros((14, 14)).astype(complex)
    lin = len(II)
    for x in range(lin):
        if Ltype[x] == 1:
            y1[II[x], JJ[x]] = 1/(R[x]+1j*X[x])
            y1[JJ[x], II[x]] = y1[II[x], JJ[x]]
            y3[II[x], JJ[x]] = 1j*B[x]
            y3[JJ[x], II[x]] = 1j*B[x]
        else:
            y1[II[x], JJ[x]] = 1/(R[x]+1j*X[x]*K[x])
            y1[JJ[x], II[x]] = y1[II[x], JJ[x]]
            y2[II[x], JJ[x]] = (1-K[x])/((R[x]+1j*X[x])* (K[x] **2))
            y2[JJ[x], II[x]] = (K[x]-1)/((R[x]+1j*X[x])*K[x])
    Y = np.zeros((14, 14)).astype(complex)
    for x in range(14):
        Y[x, x] = np.sum(y1[x]) + np.sum(y2[x]) + np.sum(y3[x])+1j*Qsh[x]

    Y -= y1
    G, B = np.real(Y), np.imag(Y)

    U = Vset.astype(complex)
    e, f = np.real(U), np.imag(U)

    D = np.ones((26,))
    Ps = (Pgen - Pload) / baseMVA
    Qs = (Qgen - Qload) / baseMVA

    N = 0;
    Jacbi = np.zeros((26, 26))
    while max(abs(D)) > 0.000001:
        for x in range(1, 14):
            a, b = np.dot(G[x], e), np.dot(B[x], f)
            c, d = np.dot(G[x], f), np.dot(B[x], e)
            if Vtype[x] == 1:
                D[2*x-2] = Ps[x]-e[x]*(a-b) - f[x]*(c + d)
                D[2*x-1] = Qs[x]-f[x]*(a-b) + e[x]*(c + d)
            else:
                D[2*x-2] = Ps[x]-e[x]*(a-b) - f[x]*(c + d)
                D[2*x-1] = Vset[x]**2-(e[x]**2 + f[x]**2)

        for I in range(1, 14):
            for J in range(1, 14):
                if I != J:
                    Jacbi[2*I-2, 2*J-2] = B[I, J]*e[I] - G[I, J]*f[I] 
                    Jacbi[2*I-2, 2*J-1] = -(G[I, J]*e[I] + B[I, J]*f[I]) 

                    if Vtype[I] == 1:
                        Jacbi[2*I-1, 2*J-2] = G[I, J]*e[I] + B[I, J]*f[I] 
                        Jacbi[2*I-1, 2*J-1] = B[I, J]*e[I] - G[I, J]*f[I] 
                    else:
                        Jacbi[2*I-1, 2*J-2] = 0; 
                        Jacbi[2*I-1, 2*J-1] = 0; 
                else:
                    a, b = np.dot(G[I], f), np.dot(B[I], e)
                    c, d = np.dot(G[I], e), np.dot(B[I], f)
                    Jacbi[2*I-2, 2*J-2]= -(a + b) + B[I, I]*e[I] - G[I,I]*f[I] 
                    Jacbi[2*I-2, 2*J-1] = -(c-d) - G[I, I]*e[I] - B[I, I]*f[I] 
                    if Vtype[I] == 1:
                        Jacbi[2*I-1, 2*J-2] = -(c-d) + G[I, I]*e[I] + B[I, I]*f[I]
                        Jacbi[2*I-1, 2*J-1] = (a+b) + B[I, I]*e[I] - G[I, I]*f[I] 
                    else:
                        Jacbi[2*I-1,2*J-2] = -2*f[I] 
                        Jacbi[2*I-1,2*J-1] = -2*e[I]

        Deta = -np.dot(np.linalg.inv(Jacbi), D)
        for x in range(1, 14):
            f[x] += Deta[(2*x-2)] 
            e[x] += Deta[(2*x-1)] 
        U = e + 1j*f
        N += 1

    N -= 1
    S0  = U[0] * np.dot(np.conj(Y[0,:]), np.conj(U))
    S1 = np.zeros((20,)).astype(complex)
    S2 = np.zeros((20,)).astype(complex)
    for x in range(20):
        S1[x] = U[II[x]] * (np.conj(U[II[x]]) * (np.conj(y2[II[x], 0]) + y3[II[x], 0] + 1j*Qsh[II[x]]) + (np.conj(U[II[x]]) - np.conj(U[JJ[x]])) * np.conj(y1[II[x],JJ[x]]))
        S2[x] = U[JJ[x]] * (np.conj(U[JJ[x]]) * (np.conj(y2[JJ[x], 0]) + y3[JJ[x], 0] + 1j*Qsh[JJ[x]]) + (np.conj(U[JJ[x]]) - np.conj(U[II[x]])) * np.conj(y1[II[x],JJ[x]]))
    detaS  = S1 + S2;
    Vabs   = np.abs(U)
    PL     = np.sum(np.real(detaS))
    Vsm    = 1 / np.min(np.linalg.svd(Jacbi)[1])
    phi    = (np.abs(Vabs[Data1[:,1] == 1]-1)-0.05) / Vabs[Data1[:,1]==1]
    DeltaV = np.sum(phi[phi>0])
    return np.array([PL, DeltaV, Vsm])


def train_net(X, T, W1, W2, nEpoch):
    n = np.shape(X)[0]
    for epoch in range(nEpoch):
        Z, Y = predict_net(X, W1, W2)
        P = (Z-T) * Z * (1-Z)
        Q  = np.dot(P, W2[1:, :].T) * (1-Y**2)
        D1 = 0
        D2 = 0
        for i in range(n):
            D2 = D2 + np.dot(np.c_[1, Y[i]].T, P[i:i+1])
            D1 = D1 + np.dot(np.c_[1, X[i:i+1, :]].T, Q[i:i+1])
        W1 -= D1/n
        W2 -= D2/n
    return W1, W2

def predict_net(X, W1, W2):
    n = np.shape(X)[0]
    Y = 1 - 2 / (1+np.exp(np.dot(2*np.c_[np.ones((n, 1)), X], W1)))
    Z = 1 / (1+np.exp(-np.dot(np.c_[np.ones((np.shape(Y)[0],1)), Y], W2)))
    return Z, Y 
