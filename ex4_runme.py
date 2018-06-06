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
from decision_tree import DecisionTree, Node
import ex4_tools
from bagging import Bagging
from datetime import datetime

def getSynData():
    files = ["X_test.txt", "X_train.txt","X_val.txt", "y_test.txt",
             "y_train.txt", "y_val.txt"]
    return [np.loadtxt("SynData/" + file) for file in files]

def getSpamData():
    return np.loadtxt("SpamData/spam.data")

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
    D = [3, 6, 8, 10, 12]
    test_err, validation_err = list(), list()

    for d in D:
        DT = DecisionTree(d)
        DT.train(X_train,Y_train)
        # ex4_tools.decision_boundaries(DT,X_train,Y_train,"decision tree, d: "+ str(d))
        test_err.append(DT.error(X_test,Y_test))
        validation_err.append(DT.error(X_val,Y_val))
        # time = str(datetime.now().strftime('%Y-%m-%d%H.%M.%S'))
        # ex4_tools.view_dtree(DT, filename=time)

    graph_plot("D","decision trees",D,test_err,
               "test err", validation_err,"validation err","Q4")
    return

def split(data):
    data_sets = [0] * len(data)
    for i in range(len(data)):
        test = np.vstack([data[j] for j in range(len(data)) if j!=i])
        validation = data[i]
        data_sets[i] = np.array([test, validation])
    return data_sets

def Q5(): # spam data

    SpamData = getSpamData()
    # changing the labels from 0 to -1, because this is how the classifiers work.
    SpamData[:, -1] [SpamData[:, -1] == 0] = -1
    # Shuffling the data
    np.random.shuffle(SpamData)
    np.random.shuffle(SpamData)
    mSpam = len(SpamData)
    inds_vault = np.random.choice(mSpam, 1536, replace=False)
    inds_train = np.array([i for i in range(mSpam) if i not in inds_vault])

    data_train = SpamData[inds_train]
    data_vault = SpamData[inds_vault]
    m_train = len(data_train)
    # get a folds as a tuple (train, validation,i)
    folds = np.split(data_train, [int(m_train /5), 2*int(m_train /5), 3*int(m_train /5), 4*int(m_train /5)])
    data_5_cv = split(folds)

    T = [5,50,100,200,500,1000]
    D = [5,8,10,12,15,18]

    err_DT, err_AdaB = [0] * 6, [0] * 6

    for i in range(5):
        print("*******i: ", i,  "  *********")
        s = data_5_cv[i][0].shape[1]
        X_train, y_train  = data_5_cv[i][0][:, 0:s - 1],data_5_cv[i][0][:, s - 1:s]
        X_validation, y_validation = data_5_cv[i][1][:, 0:s - 1],data_5_cv[i][1][:, s - 1:s]
        y_train = y_train.reshape((-1,))
        y_validation = y_validation.reshape((-1,))

        # train decision tree
        for d in D:
            dt = DecisionTree(d)
            dt.train(X_train,y_train)
            e = dt.error(X_validation,y_validation)
            print("d:",str(d), " e: ",e)
            err_DT[i] +=dt.error(X_validation,y_validation)

        #train AdaBoost
        for t in T:
            adaB = adaboost.AdaBoost(ex4_tools.DecisionStump,t)
            adaB.train(X_train,y_train)
            e = adaB.error(X_validation,y_validation)
            err_AdaB[i] += e
            print("t:",str(t), " e: ",e)


    err_DT, err_AdaB = np.array([i/5 for i in err_DT]), np.array([i/5 for i in err_AdaB])
    print("err_DT: ",err_DT)
    print("err_ADAB: ",err_AdaB)

    # checking optimal value for the data vault
    X ,y = SpamData[:,0:57],SpamData[:,57:58].reshape((len(SpamData),))
    X_test, y_test = data_vault[:,0:57],data_vault[:,57:58].reshape((1536,))
    dt = DecisionTree(15)
    dt.train(X, y)
    e = dt.error(X_test,y_test)
    print( "DT: e -  ", e)

    adaB = adaboost.AdaBoost(ex4_tools.DecisionStump, 100)
    adaB.train(X, y)
    e = adaB.error(X_test,y_test)
    print( "adaBoost: e =  ", e)

    return

def bagging():
    # B values as hyperparameters
    B = [1,2,4,6,8,10,12,14,16,18,20]
    syn_data = getSynData()
    X_test, X_train, X_val = syn_data[0], syn_data[1], syn_data[2]
    Y_test, Y_train, Y_val = syn_data[3], syn_data[4], syn_data[5]


    test_err, validation_err = list(), list()

    for b in B:
        b_cls = Bagging(DecisionTree,b)
        b_cls.train(X_train, Y_train)

        test_err.append(b_cls.error(X_test, Y_test))
        validation_err.append(b_cls.error(X_val, Y_val))

    graph_plot("b", "bagging with decision trees", B, test_err,
               "test err", validation_err, "validation err", "Bagging")

if __name__ == '__main__':
    Q3()
    Q4()
    bagging()
    Q5()




    # ************************ SAVED THE RESULTS SINCE IT TOOK A LOT OF TIME TO COMPUTE ***************
    # err_adaB[0] = (0.0978792822186 + 0.112561174551 +0.145187601958+0.110929853181+0.0913539967374)/5
    # err_adaB[1] = (0.0587275693312 + 0.0717781402936 + 0.0717781402936 + 0.0652528548124 +  0.068515497553)/5
    # err_adaB[2] = (0.0489396411093 + 0.0717781402936 + 0.0603588907015 +  0.0636215334421 + 0.047308319739)/5
    # err_adaB[3] = (0.0456769983687 + 0.0652528548124 + 0.0587275693312 + 0.0668841761827 + 0.0554649265905)/5
    # err_adaB[4] = (0.0522022838499+ 0.0701468189233 +   0.0587275693312  + 0.0766721044046 + 0.047308319739)/5
    # err_adaB[5] = (0.0538336052202 +0.0750407830343 + 0.0587275693312 + 0.0734094616639+ 0.0587275693312)/5
    #
    # err_DT[0] = (0.31484502447 +0.342577487765+0.342577487765+0.323001631321+0.36867862969)/5
    # err_DT[1] = (0.334420880914 +  0.309951060359 + 0.340946166395 + 0.3132137031 + 0.345840130506)/5
    # err_DT[2] = (0.308319738989 + 0.306688417618 +  0.301794453507 + 0.305057096248 + 0.319738988581)/5
    # err_DT[3] = (0.292006525285 + 0.296900489396 + 0.277324632953 + 0.292006525285 +0.301794453507)/5
    # err_DT[4] = (0.270799347471 +0.246329526917+0.265905383361+0.26101141925+0.247960848287)/5
    # err_DT[5] = (0.269168026101+0.249592169657+ 0.274061990212+0.293637846656+ 0.272430668842)/5
    # *******i:  0   *********
    # d: 5  e:  0.31484502447
    # d: 8  e:  0.334420880914
    # d: 10  e:  0.308319738989
    # d: 12  e:  0.292006525285
    # d: 15  e:  0.270799347471
    # d: 18  e:  0.272430668842
    # t: 5  e:  0.0978792822186
    # t: 50  e:  0.0587275693312
    # t: 100  e:  0.047308319739
    # t: 200  e:  0.0456769983687
    # t: 500  e:  0.0522022838499
    # t: 1000  e:  0.0538336052202
    # *******i:  1   *********
    # d: 5  e:  0.342577487765
    # d: 8  e:  0.309951060359
    # d: 10  e:  0.306688417618
    # d: 12  e:  0.296900489396
    # d: 15  e:  0.246329526917
    # d: 18  e:  0.293637846656
    # t: 5  e:  0.112561174551
    # t: 50  e:  0.0717781402936
    # t: 100  e:  0.0636215334421
    # t: 200  e:  0.0652528548124
    # t: 500  e:  0.0701468189233
    # t: 1000  e:  0.0750407830343
    # *******i:  2   *********
    # d: 5  e:  0.342577487765
    # d: 8  e:  0.340946166395
    # d: 10  e:  0.301794453507
    # d: 12  e:  0.277324632953
    # d: 15  e:  0.265905383361
    # d: 18  e:  0.274061990212
    # t: 5  e:  0.145187601958
    # t: 50  e:  0.0717781402936
    # t: 100  e:  0.0603588907015
    # t: 200  e:  0.0587275693312
    # t: 500  e:  0.0587275693312
    # t: 1000  e:  0.0587275693312
    # *******i:  3   *********
    # d: 5  e:  0.323001631321
    # d: 8  e:  0.3132137031
    # d: 10  e:  0.305057096248
    # d: 12  e:  0.292006525285
    # d: 15  e:  0.26101141925
    # d: 18  e:  0.249592169657
    # t: 5  e:  0.110929853181
    # t: 50  e:  0.0652528548124
    # t: 100  e:  0.0717781402936
    # t: 200  e:  0.0668841761827
    # t: 500  e:  0.0766721044046
    # t: 1000  e:  0.0734094616639
    # *******i:  4   *********
    # d: 5  e:  0.36867862969
    # d: 8  e:  0.345840130506
    # d: 10  e:  0.319738988581
    # d: 12  e:  0.301794453507
    # d: 15  e:  0.247960848287
    # d: 18  e:  0.269168026101
    # t: 5  e:  0.0913539967374
    # t: 50  e:  0.068515497553
    # t: 100  e:  0.0489396411093
    # t: 200  e:  0.0554649265905
    # t: 500  e:  0.047308319739
    # t: 1000  e:  0.0587275693312
