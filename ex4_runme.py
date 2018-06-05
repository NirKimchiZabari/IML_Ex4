"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Nir Zabari
Date: May, 2018
"""
import matplotlib.pyplot as plt
import numpy as np
import adaboost
import decision_tree
import ex4_tools
from datetime import datetime

def getSynData():
    files = ["X_test.txt", "X_train.txt","X_val.txt", "y_test.txt",
             "y_train.txt", "y_val.txt"]
    return [np.loadtxt("SynData/" + file) for file in files]

def graph_plot(x_desc,y_desc,x,y_1,y_1_desc,y_2,y_2_desc,title,y_3 = None,y_3_desc = None):
    plt.plot(x,y_1,linewidth = 1.5,label=y_1_desc,color = 'r')
    plt.plot(x,y_2,linewidth = 1.5,label=y_2_desc,color = 'g')
    if y_3 != None:
        plt.plot(x,y_3,linewidth = 1.5,label=y_3_desc,color = 'k')

    plt.title(title)
    plt.xlabel(x_desc)
    plt.ylabel(y_desc)
    plt.legend()
    plt.show()

def Q3(): # AdaBoost
    syn_data = getSynData()
    X_test,X_train,X_val = syn_data[0],syn_data[1],syn_data[2]
    Y_test,Y_train,Y_val = syn_data[3] ,syn_data[4] ,syn_data[5]

    T = [5*(i+1) for i in range(20)] + [200]
    test_err, validation_err = list(), list()

    for t in T:
        AdaB = adaboost.AdaBoost(ex4_tools.DecisionStump,t)
        AdaB.train(X_train,Y_train)

        test_err.append(AdaB.error(X_test,Y_test))
        validation_err.append(AdaB.error(X_val,Y_val))

    graph_plot("T","error of adaBoost",T,test_err,
               "test err", validation_err,"validation err","Q3")
    #
    return

def Q4(): # decision trees
    syn_data = getSynData()
    X_test, X_train, X_val = syn_data[0], syn_data[1], syn_data[2]
    Y_test, Y_train, Y_val = syn_data[3], syn_data[4], syn_data[5]
    # D = [3, 6, 8, 10, 12]
    test_err, validation_err = list(), list()
    D = [3]

    DT = decision_tree.DecisionTree(3)
    DT.root = decision_tree.Node()
    DT.root.theta = 3
    DT.root.feature = 3


    for d in D:
        DT = decision_tree.DecisionTree(d)
        DT.train(X_train,Y_train)
        # ex4_tools.decision_boundaries()

        time = str(datetime.now().strftime('%Y-%m-%d%H.%M.%S'))

        ex4_tools.view_dtree(DT,filename=time)

        test_err.append(DT.error(X_test,Y_test))
        validation_err.append(DT.error(X_val,Y_val))

    return

def Q5(): # spam data
    # TODO - implement this function
    return

def polygon_approxtimation(Points):
    sum = 0
    for i in range(len(Points)-1):
        sum += Points[i][0] * Points[i+1][1] - Points[i][1] * Points[i+1][0]
    return sum/2

if __name__ == '__main__':
    # Q3()
    Q4()
    # Q5()