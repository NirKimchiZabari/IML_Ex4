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
        self.CART(X,y,None,self.max_depth)

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
        # I will use the implementation as given in the Decision stump,
        #since it implements very similar functionality.
        self.root = Node()
        self.root.leaf = False
        self.root.right, self.root.left  = Node(), Node()
        self.cart_helper(X, y, self.root, 1)
        print("hello")
        return self.root

    def cart_helper(self,X , y, cur, depth):
        if (depth == self.max_depth):
            cur.leaf = True
            return
        theta, j, s = 0, 0, 0
        print(type(X))
        print(X.shape)
        if len(X) == 0:
            return
        m, d = X.shape

        D = np.array([1/m] * m)
        F, J, theta = [0]*2, [0]*2, [0]*2
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
        # This is the requested cut
        theta, j, s = theta[b], J[b], 2*b-1
        cur.feature, cur.theta = j, theta

        # We don't want to choose the feature j again.
        # X = np.delete(X,j,1)
        cur.leaf = False
        cur.left , cur.right = Node(), Node()
        X_left, y_left ,X_right , y_right= [],[],[],[]
        for i in range(m):
            if X[i][j] > theta:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        print("called left. depth: ",depth)
        self.cart_helper(np.array(X_left),np.array(y_left),cur.left,depth+1)
        print("called right. depth: ",depth)

        self.cart_helper(np.array(X_right),np.array(y_right),cur.right,depth+1)

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
        ans = self.predict(X)
        return np.sum(np.logical_not(np.equal(ans, y))) / len(X)
