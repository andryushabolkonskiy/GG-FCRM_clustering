import argparse
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster._kmeans import _kmeans_plusplus
from numpy.random import RandomState
from fuzzylab import *
import math


class GathGeva:
    def __init__(self, path, c, m, seed, iter_max, epsilon):
        self.data = np.genfromtxt(path, delimiter = ',', skip_header = 1)
        self.n, self.p = self.data.shape
        self.c = c
        self.m = m
        self.seed = seed
        self.iter_max = iter_max
        self.epsilon = epsilon
        self.u = self.imembrand()
        self.start_gg()

    def imembrand(self):
        u = np.random.randint(low=1, high=100, size=(self.n, self.c))
        u = u / u.sum(axis=1).reshape(-1, 1)
        row_sums =  u.sum(axis=1).reshape(-1, 1)
        u[:, [0]] = u[:, [0]] + np.where(row_sums == 1, row_sums * 0, 1 - row_sums)
        return u

    def start_gg(self):
        x_squared_norms = np.einsum('ij,ij->i', self.data, self.data)
        self.v, indices = _kmeans_plusplus(self.data, self.c, x_squared_norms,
                                           random_state=RandomState(self.seed), n_local_trials=None)
        d = np.zeros((self.n, self.c))
        prevu = self.u + 2*self.epsilon
        self.iter = 0

        while (self.iter < self.iter_max) and (np.linalg.norm(prevu - self.u) > self.epsilon):
            self.iter += 1
            prevd, prevu, prevv = d, self.u, self.v
            v = (self.u ** self.m).T @ (np.divide(self.data, (self.u ** self.m).sum(axis=1).reshape(-1, 1)))
            self.all_aj = []
            for j in range(self.c):
                aj = np.zeros((self.p, self.p))
                for i in range(self.n):
                    z = self.data[i, :] - v[j, :]
                    z = z.reshape(-1, 1)
                    to_add = np.full(aj.shape, (self.u[i, j] ** self.m) * (z @ z.T))
                    aj = aj + to_add
                aj = aj / np.sum(self.u[:, j] ** self.m)
                self.all_aj.append(aj)
                alpha_j = np.sum(self.u[:, j] ** self.m) / np.sum(self.u ** self.m)
                for i in range(self.n):
                    z = self.data[i, :] - self.v[j, :]
                    d[i, j] = (
                            np.sqrt(np.linalg.det(aj)) / alpha_j
                            * np.exp(1/2 * z.T @ np.linalg.pinv(aj) @ z)
                    )
            for i in range(self.n):
                V = np.sum(np.divide(np.ones(d[i, :].shape), d[i, :]) ** (1 / (self.m - 1)))
                A = V * d[i, :] ** (1 / (self.m - 1))
                self.u[i, :] = np.divide(np.ones(A.shape), A)
            if np.any(np.isnan(self.u)) or np.any(np.isinf(self.u)):
                d, self.u, self.v = prevd, prevu, prevv
            for i in range(self.n):
                for j in range(self.c):
                    if self.u[i, j] < 0:
                        self.u[i, j] = 0
                    elif self.u[i, j] > 1:
                        self.u[i, j] = 1
        clabels = np.max(self.u, axis=1)


class TakagiSugeno:
    def __init__(self, path, c, m, seed, iter_max, epsilon):
        self.clustering = GathGeva(path, c, m, seed, iter_max, epsilon)
        self.run_TS()

    def run_TS(self):
        self.params = []
        p = self.clustering.p
        for i in range(self.clustering.c):
            F_ii = np.delete(self.clustering.all_aj[i], p-1, axis=1)
            F_ii = np.delete(F_ii, p-1, axis=0)
            eigenValue, eigenVector = np.linalg.eig(F_ii)
            for j in range(p-1):
                sigma = eigenValue[j] * eigenValue[j]
                v_tmp = np.delete(self.clustering.v[i], p-1)
                v_tmp = v_tmp.reshape(-1, 1)
                t_t = eigenVector[j].reshape(1, -1)
                vii = t_t @ v_tmp
                self.params.append([vii[0][0], sigma])

        for j in range(p-1):
            for i in range(self.clustering.c):
                a = self.params[j+(p-1)*i][0] + 4 * self.params[j+(p-1)*i][1]
                b = self.params[j+(p-1)*i][0] - 4 * self.params[j+(p-1)*i][1]
                x = np.linspace(a, b, 401)
                y = gaussmf(x, [self.params[j+(p-1)*i][1], self.params[j+(p-1)*i][0]])
                plt.plot(x, y)
            plt.show()



if __name__ == '__main__':
##    F = input("input data file name: ")
##    C = int(input("input clusters ammount (integer < data lenght): "))
##    M = float(input("input m (float > 1): "))
##    E = float(input("input epsilon (float): "))

    F = "data_02.csv"
    C = 2
    M = 2
    E = 1e-09

    TS = TakagiSugeno(F, C, M, 42, 1000, E)
