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

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """


    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
