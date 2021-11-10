import matplotlib.pyplot as plt
from fuzzylab import *
import numpy as np
import math
import random

class Fcrm:
    def __init__(self, path, c, m, epsilon):
        self.data = np.genfromtxt(path, delimiter = ',', skip_header = 1)
        self.number, self.dim = self.data.shape
        self.c = c
        self.m = m
        self.epsilon = epsilon
        self.U = np.zeros(shape = (self.c, self.number))
        self.E = np.zeros(shape = (self.c, self.number))
        self.fill_random_matrix(self.U)
        self.clusters = np.array([[0.0 for i in range(self.dim)] for j in range(self.c)])
        self.start_fcrm()

    def fill_random_matrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                summ = 0
                for k in range(i):
                    summ += matrix[k][j]
                if (i != len(matrix) - 1):
                    matrix[i][j] = random.uniform(0, 1-summ)
                else:
                    matrix[i][j] = max(1-summ, 0)
        
    def start_fcrm(self):
        self.iter = 0
        while (self.iter == 0) or ((error > self.epsilon) and (self.iter < 1000)):
            U_old = np.copy(self.U)
            self.update_clusters()
            self.update_matrix()
            error = np.linalg.norm(U_old - self.U)
            self.iter += 1

    def update_clusters(self):
        X_data = self.data[:, :-1]
        X = np.ones(shape = (self.number, self.dim))
        for i in range(len(X)):
            X[i][1:] = X_data[i]
        Y = self.data[:, -1].reshape(1, self.number)
        for i in range(self.c):
            D = np.zeros(shape = (self.number, self.number))
            for k in range(self.number):
                D[k, k] = pow(self.U[i, k], self.m)
            mtr1 = (X.T).dot(D).dot(X)
            mtr2 = np.linalg.pinv(mtr1)
            self.clusters[i] = np.dot(mtr2, ((X.T).dot(D).dot(Y.T))).reshape(self.dim,)

    def update_matrix(self):
        self.make_error_matrix()
        for k in range(self.number):
            den = np.zeros(shape=(self.c, 1))
            for i in range(self.c):
                a = self.calculate_denominator(i, k)
                if a != 0:
                    den[i, 0] = float(1/a)
                else:
                    den[i, 0] = 0
            self.U[:, k] = den[:, 0]

    def calculate_denominator(self, i, k):
        summ = 0.0
        for j in range(self.c):
            summ = summ + math.pow(self.E[i, k]/self.E[j, k], 1/(self.m - 1))
        return summ

    def make_error_matrix(self):
        for i in range(self.c):
            for j in range(self.number):
                params = np.ones(self.dim)
                params[1:] = self.data[j, :-1]
                self.E[i, j] = self.calculate_error(i, self.data[j, -1], params)

    def calculate_error(self, i, Y, X):
        return math.pow(Y - (np.dot(self.clusters[i], X)), 2)



class TakagiSugeno:
    def __init__(self, path, c, m, epsilon):
        self.clustering = Fcrm(path, c, m, epsilon)
        self.calculate_params()
        self.draw_all()

    def calculate_P1(self, i, j):
        s1, s2 = 0.0, 0.0
        for k in range(self.clustering.number):
            s1 += self.clustering.U[i][k] * self.clustering.data[k][j]
            s2 += self.clustering.U[i][k]
        return s1/s2

    def calculate_P2(self, i, j, p1):
        s1, s2 = 0.0, 0.0
        for k in range(self.clustering.number):
            s1 += self.clustering.U[i][k] * pow(self.clustering.data[k][j] - p1, 2)
            s2 += self.clustering.U[i][k]
        return math.sqrt(s1/s2)

    def calculate_params(self):
        self.params = []
        for j in range(self.clustering.dim - 1):
            for i in range(self.clustering.c):
                P1 = self.calculate_P1(i, j)
                P2 = self.calculate_P2(i, j, P1)
                self.params.append([P1, P2])
            
    def draw_all(self):
        for j in range(self.clustering.dim - 1):
            for i in range(self.clustering.c):
                P1 = self.params[j*self.clustering.c + i][0]
                P2 = self.params[j*self.clustering.c + i][1]
                a = P1 + 4*P2
                b = P1 - 4*P2        
                x = np.linspace(a, b, 401)
                y = gaussmf(x, [P2, P1])
                plt.title('X'+str(j+1))
                plt.plot(x, y, label = 'Cluster: '+str(i+1))
                plt.legend()
            plt.show()



if __name__ == '__main__':
    F = input("input data file name: ")
    C = int(input("input clusters ammount (integer < data lenght): "))
    M = float(input("input m (float > 1): "))
    E = float(input("input epsilon (float): "))

##    F = "data_02.csv"
##    C = 2
##    M = 2
##    E = 0.00005

    TS = TakagiSugeno(F, C, M, E)
