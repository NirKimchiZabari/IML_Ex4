"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np
import math

class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m = len(X) # I assume X is a list of examples.
        D = np.array([1/m for _ in range(m)]).reshape(m, 1)  #initialize the weights vector

        for t in range(self.T):
            h = self.WL(D,X,y)
            epsilon_t = self.calculate_epsilon_t(X,y,D,h)
            w_t = (1/2)* math.log((1/epsilon_t) - 1)

            # saving the new predictor
            self.h[t] = h
            self.w[t] = w_t

            # updating D_i for iteration t+1
            p = h.predict(X).reshape((-1, 1))
            d = (D.T).dot(np.exp(-w_t * y * p))[0][0]
            D = D * np.exp(-w_t * y * p)/(d)


    def calculate_epsilon_t(self,X,y,D,h):
        """
        calculate epsilon_t from the algorithm, where X is the samples, y
        is the true labels, D is the weights vector, and h is the weak learner.
        """
        # epsilon_t = 0
        # for t in range(len(X)):
        #     if not h.predict(X[t]) == y[t]:
        #         epsilon_t += D[t] # D[t]*1 from the psuedo code, it's obvious
        # return epsilon_t

        return np.sum(D * [y!= h.predict(X).reshape(-1,1)])


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        T_pred = [(self.h[i]).predict(X) for i in range(self.T)]

        res = np.zeros(len(X)).reshape(len(X),1)
        for i in range(len(X)):
                for t in range(self.T):
                    res += self.w[t] * T_pred[t][i]
        return np.sign(res)


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        if len(X) != len(y):
            raise Exception("Different number of labels and samples")

        res = self.predict(X)
        return sum([1 if y[i] == res[i] else 0 for i in range(len(y))])

