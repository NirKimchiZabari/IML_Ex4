"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the decision tree classifier with real-values features.
Training algorithm: CART

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy as np

class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self,leaf = True,left = None,right = None,samples = 0,feature = None,theta = 0.5,misclassification = 0,label = None):
        """
        Parameters
        ----------
        leaf : True if the node is a leaf, False otherwise
        left : left child
        right : right child
        samples : number of training samples that got to this node
        feature : a coordinate j in [d], where d is the dimension of x (only for internal nodes)
        theta : threshold over self.feature (only for internal nodes)
        label : the label of the node, if it is a leaf
        """
        self.leaf = leaf
        self.left = left
        self.right = right
        self.samples = samples
        self.feature = feature
        self.theta = theta
        self.label = label


class DecisionTree(object):
    """ A decision tree for binary classification.
        max_depth - the maximum depth allowed for a node in this tree.
        Training method: CART
    """

    def __init__(self,max_depth):
        self.root = None
        self.max_depth = max_depth

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """

    def CART(self,X, y, A, depth):
        """
        Gorw a decision tree with the CART method ()

        Parameters
        ----------
        X, y : sample
        A : array of d*m real features, A[j,:] row corresponds to thresholds over x_j
        depth : current depth of the tree

        Returns
        -------
        node : an instance of the class Node (can be either a root of a subtree or a leaf)
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
