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
import ex4_tools

class Node(object):
    """ A node in a real-valued decision tree.
        Set all the attributes properly for viewing the tree after training.
    """
    def __init__(self, leaf = True, left = None, right = None, samples = 0,
                 feature = None, theta = 0.5, misclassification = 0,
                 label = None):
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
        self.num_features = 0

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        if len(X) == 0:
            raise Exception("number of train sample should be positive.")
        if len(X) != len(y):
            raise Exception("different num of samples")

        self.CART(X,y,None,self.max_depth)

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        if len(X) == 0:
            raise Exception("you can't train without examples.")
        A = list()
        for i in range(len(X[0])):
            A.append(np.sort(np.unique(X[:,i])))

        self.root = self.CART(X, y, np.array(A), 0)




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
        self.root = Node()
        self.CART_helper(self.root,X,y,A,depth)
        return self.root

    def get_best_split(self, X,y,D = []):
        m, d = X.shape
        F, J, theta = [0]*2, [0]*2, [0]*2
        D = np.array([1/m] * m)
        for b in [0,1]:
            s = 2*b - 1
            F[b], theta[b], J[b] = D[y==s].sum(), X[:,0].min()-1, 0
            for j in range(d):  # go over all features
                ind = np.argsort(X[:, j])
                Xj = np.sort(X[:, j])  # sort by coordinate j
                Xj = np.hstack([Xj,Xj.max()+1])
                f = D[y==s].sum()
                for i in range(m): # check thresholds over Xj for improvement
                    f -= s*y[ind[i]]*D[ind[i]]
                    if f < F[b] and Xj[i] != Xj[i+1]:
                        F[b], J[b], theta[b] = f, j, (Xj[i]+Xj[i+1])/2
        b = np.argmin(F)
        return theta[b], J[b], 2*b-1


    def split_data(self, X, y, theta, j, s):
        X_left,y_left,X_right,y_right = None,None,None,None
        # if s == 1:
        #     X_left = X[X[:,j] >= theta]
        #     y_left = y[X[:,j] >= theta]
        #     X_right = X[X[:,j] < theta]
           #     y_right = y[X[:,j] < theta]
           # else:
        m, d = np.shape(X)
        X_left = X[X[:, j] <= theta]
        y_left = y[X[:, j] <= theta]
        X_right = X[X[:, j] > theta]
        y_right = y[X[:, j] > theta]

        if np.shape(X_left) == (0,) or np.shape(X_left) == (m, d):
            X_left = X[:int(m / 2), :]
            y_left = y[:int(m / 2)]
            X_right = X[int(m / 2):, :]
            y_right = y[int(m / 2):]
        return X_left,y_left,X_right,y_right


    def check_for_zeros(self, label_left, label_right, feature, theta, X, y):
        m, d = np.shape(X)
        X_left = X[X[:, feature] <= theta]
        y_left = y[X[:, feature] <= theta]
        X_right = X[X[:, feature] > theta]
        y_right = y[X[:, feature] > theta]

        if np.shape(X_left) == (0,) or np.shape(X_left) == (m,d):
            X_left = X[:int(m/2),:]
            y_left = y[:int(m/2)]
            X_right = X[int(m/2):,:]
            y_right = y[int(m/2):]
        if np.shape(X_left) == (0,):
            label_left = label_right
        if np.shape(X_left) == (m,d):
            label_right = label_left
        return X_left, y_left, X_right, y_right, label_left, \
                label_right

    def CART_helper(self, root, X, y, A, depth):
    #     find best split
        if len(X) <= 1 :
            root.leaf = True
            return
        theta,j,s = self.get_best_split(X,y)

        X_left,y_left,X_right,y_right = self.split_data(X,y,theta,j,s)

    #     add the nodes
        root.samples = len(X)
        root.feature = j
        root.theta = theta
        root.leaf = False

        root.left, root.right = Node(),Node()
        root.left.leaf, root.right.leaf= False, False
        root.left.label, root.right.label = s, (-1 * s)

        if len(X) == len(X_left):
            root.right.label = root.label.left
        if len(X_left) == 0:
            root.left.label = root.right.left

                #     split the data
    # if we are in a good depth, continue recursively
        if depth < self.max_depth:
            self.CART_helper(root.left,X_left,y_left,A,depth+1)
            self.CART_helper(root.right,X_right,y_right,A,depth+1)
        else:
            root.right,root.left,root.leaf = None, None, True



    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """
        return np.array([self.predict_single(x) for x in X])

    def predict_single(self,x):
        # todo: to change
        # if len(x) != self.num_features:
        #     raise Exception("different number of features")
        cur = self.root
        while cur.leaf == False:
            if x[cur.feature] >= cur.theta:
                cur = cur.right
            else:
                cur = cur.left
        return cur.label

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """
        ans = self.predict(X)
        return np.sum(np.logical_not(np.equal(ans, y))) / len(X)