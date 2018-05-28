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
import pandas as ps

import ex4_tools

def getSynData():
    files = ["X_test.txt", "X_train.txt","X_val.txt", "y_test.txt",
             "y_train.txt", "y_val.txt"]
    return [ps.read_csv("SynData/"+ file,header=None,delim_whitespace=True).values for file in files]


def getAccuracy(p, t_labels):
    if len(p) != len(t_labels):
        raise Exception("different number.")
    m = len(p)
    n_err = np.sum(p==t_labels)
    return n_err/m

def graph_plot(x_desc,y_desc,x,y_1,y_1_desc,y_2,y_2_desc,title):
    plt.plot(x,y_1,linewidth = 1.5,label=y_1_desc,color = 'r')
    plt.plot(x,y_2,linewidth = 1.5,label=y_2_desc,color = 'g')
    plt.title(title)
    plt.xlabel(x_desc)
    plt.ylabel(y_desc)
    plt.legend()
    plt.show()

def Q3(): # AdaBoost
    # Get the data
    syn_data = getSynData()
    X_test,X_train,X_val = syn_data[0],syn_data[1],syn_data[2]
    Y_test,Y_train,Y_val = syn_data[3],syn_data[4],syn_data[5]

    T = [5*(i+1) for i in range(20)] + [200]
    # the error is in percentage, will be: total mistakes/total samples
    training_err, validation_err = list(), list()

    PPP = list()
    for t in T:
        AdaB = adaboost.AdaBoost(ex4_tools.DecisionStump,t)
        AdaB.train(X_train,Y_train)

        # predict_test = AdaB.predict(X_test)
        # predict_val = AdaB.predict(X_val)
        # PPP.append(predict_test)

        training_err.append(AdaB.error(X_test,Y_test)/len(X_test))
        validation_err.append(AdaB.error(X_val,Y_val)/len(X_val))

    graph_plot("T","error of adaBoost",T,training_err,
               "training err", validation_err,"validation err","Q3")

    return

def Q4(): # decision trees
    # TODO - implement this function
    return

def Q5(): # spam data
    # TODO - implement this function
    return

def test():
    return [1,2]

if __name__ == '__main__':
    # TODO - run your code for questions 3-5
    Q3()


    # D = np.array([1/10.0 for i in range(10)])
    # X = np.array([[1,0] for i in range(10)])
    # y = np.array([1 for i in range(10)])
	#
	# test = ex4_tools.DecisionStump(D,X,y)
	# print(test.predict([1,0]))


# data =parse_file("SynData/y_test.txt")
# for i in range(len(data)):
#     print("i: ", i, "res: ",data[i])