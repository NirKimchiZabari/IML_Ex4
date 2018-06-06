"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for classification with Bagging.

Author: Yoav Wald

"""
import numpy as np

class Bagging(object):

    def __init__(self, L, B):
        """
        Parameters
        ----------
        L : the class of the base learner
        T : the number of base learners to learn
        """
        self.L = L
        self.B = B
        self.h = [None]*B     # list of base learners
        self.m = 0

    def draw_points(self, X,y):
        # Bagging is with replacements
        # we define to sample 0.75*(given_sample_size) as a hyperparameter as well
        inds_vault = np.random.choice(len(X), int(len(X)*0.75), replace=True)
        return X[inds_vault], y[inds_vault]

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        # We define m as a constant.
        for i in range(self.B):
            sampled_X,sampled_y= self.draw_points(X,y)
            L = self.L(10) # depth of the trees is 10 - hyperparameter as mentiond in the forum
            L.train(sampled_X,sampled_y)
            self.h[i] = L


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        if self.B == 1:
            return self.h[0].predict(X)
        else:
            predictions = [self.h[i].predict(X) for i in range(self.B)]
            res = np.array([0] * len(X))
            for i in range(len(X)):
                val = 0
                for b in range(self.B):
                    val +=predictions[b][i]
                if val >=0:
                    res[i] = 1
                else:
                    res[i] = -1
            return res

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        ans = self.predict(X)
        return np.sum(np.logical_not(y==ans)) / len(X)