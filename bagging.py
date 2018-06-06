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

    def draw_points(self, X):
        pass
        #todo :draw points func

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        # We define m as a constant.
        if len(X) < 3:
            self.m = 1
        else:
            self.m = int(np.math.ceil(len(X)))

        for i in range(self.B):
            sampled_X,sampled_y= self.draw_points(X)
            L = self.L(10) #depth of the trees is 10
            L.train(sampled_X,sampled_y)
            self.h[i] = L


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        predictions = [self.h[i].predict(X) for i in range(self.B)]
        sum = [0] * len(X)
        for i in range(len(X)):
            for j in range(self.B):
                sum+=predictions[j][i]
        return [1 if sum[i] >=0 else -1 for i in range(len(X))]


    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        ans = self.predict(X)
        return np.sum(np.logical_not(np.equal(ans, y))) / len(X)
