"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

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

    # def compute(self, w,y,h_x):
    #     eq = np.equal(y,h_x)
    #     loss = np.logical_not(eq).astype(int)
    #     return np.sum(w * loss)


    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m = len(y) # number of samples
        D = np.array([1/len(y) for _ in range(m)]).reshape(m,1)

        for t in range(self.T):
            self.h[t] = self.WL(D,X,y)
            pred_h_t = self.h[t].predict(X)

            equal_arr = np.array([1 if y[i] != pred_h_t[i] else 0 for i in range(m)])
            epsilon_t = np.sum([equal_arr[i] * D[i] for i in range(m)])

            self.w[t] = (np.log((1/epsilon_t) - 1))/2

            d = np.sum([D[j] * np.exp(-1 * self.w[t] * y[j] * pred_h_t[j])  for j in range(m)])
            for j in range(m):
                D[j] = (D[j] * np.exp(-1 * self.w[t] * y[j] * pred_h_t[j])) / d


        # m,d = X.shape
        # D = np.array([1/m] * m).reshape(m,1)
        # for t in range(self.T):
        #     self.h[t] = self.WL(D,X,y)
        #     h_X = self.h[t].predict(X)
        #     epsilon_t = self.compute(D,y,h_X)
        #     self.w[t] = 0.5 * np.log((1/epsilon_t) - 1)
        #     norm_factor = 0
        #     for i in range(m):
        #         norm_factor += D[i] * np.exp(-1 * self.w[t] * y[i] * self.h[t].predict([X[i]])[0])
        #     for i in range(m):
        #         D[i] = (D[i] * np.exp(-1 * self.w[t] * y[i] * self.h[t].predict([X[i]])[0])) / norm_factor


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        y_hat = list()
        for s in X:
            val = 0
            for t in range(self.T):
                val += self.w[t] * self.h[t].predict(np.array([s]))[0]
            y_hat.append(np.sign(val))
        return np.array(y_hat)


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        predictions = self.predict(X)
        m = len(y)
        num_mistakes = sum([1 if predictions[i] != y[i] else 0 for i in range(m)])
        return num_mistakes/m
