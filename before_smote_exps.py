# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:05:47 2020

@author: 4PF41LA_RS6
"""

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn.metrics import accuracy_score
from data import Datasets
import random
import ann as rna

def separate(test_percent, X, Y, S):
    random.seed(S)
    count_test = int(test_percent * X.shape[0])
    
    visited = []
    for _ in range(count_test):
        n = random.randint(0, X.shape[0] - 1)
        
        while n in visited:
            n = random.randint(0, X.shape[0] - 1)
        
        visited.append(n)
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(X.shape[0]):
        if i in visited:
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            Y_train.append(Y[i])
            X_train.append(X[i])
    
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    
    return X_train, Y_train, X_test, Y_test
        
def nb_gaussiano(X_train, Y_train, X_test, Y_test):
    nb = GaussianNB()
    Y_pred = nb.fit(X_train, Y_train).predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    return accuracy

def knn(X_train, Y_train, X_test, Y_test, k):
    n_neighbors = k
    nn = neighbors.KNeighborsClassifier(n_neighbors)
    Y_pred = nn.fit(X_train, Y_train).predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    return accuracy

def id3(X_train, Y_train, X_test, Y_test):
    clf = tree.DecisionTreeClassifier()
    Y_pred = clf.fit(X_train, Y_train).predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    
    return accuracy

def neural_net(X_train, Y_train, X_test, Y_test, lr, act_f, epochs, prt):
    # 1. contar cantidad atributos (input layer)
    atts = X_train.shape[1]
    # 2. contar cantidad de clases diferentes (output layer)
    clss = len(np.unique(Y_train))
    top = [atts, atts - 1, clss]
    Y_pred = rna.fit(X_train, Y_train, X_test, Y_test, top, lr, act_f, epochs, prt)
    accuracy = rna.accuracy_score(Y_test, Y_pred)
    
    return accuracy

dts = Datasets()

S = 2
dts.remove_data(S)

### ABALONE ###
x_abalone = dts.X_abalone
y_abalone = dts.Y_abalone
x_abalone_r = dts.X_rem_abalone
y_abalone_r = dts.Y_rem_abalone

X_train_abalone, Y_train_abalone, X_test_abalone, Y_test_abalone = separate(0.3, x_abalone, y_abalone, S)
X_train_r_abalone, Y_train_r_abalone, X_test_r_abalone, Y_test_r_abalone = separate(0.3, x_abalone_r, y_abalone_r, S)
print("Abalone original:")
print("NBG:", nb_gaussiano(X_train_abalone, Y_train_abalone, X_test_abalone, Y_test_abalone))
print("KNN:", knn(X_train_abalone, Y_train_abalone, X_test_abalone, Y_test_abalone, 15))
print("ID3:", id3(X_train_abalone, Y_train_abalone, X_test_abalone, Y_test_abalone))
print("ANN:", neural_net(X_train_abalone, Y_train_abalone, X_test_abalone, Y_test_abalone, 0.0001, sigmoidal, 10000, prt = False))
print("Abalone rem:")
print("NBG:", nb_gaussiano(X_train_r_abalone, Y_train_r_abalone, X_test_r_abalone, Y_test_r_abalone))
print("KNN:", knn(X_train_r_abalone, Y_train_r_abalone, X_test_r_abalone, Y_test_r_abalone, 15))
print("ID3:", id3(X_train_r_abalone, Y_train_r_abalone, X_test_r_abalone, Y_test_r_abalone))
print("ANN:", neural_net(X_train_r_abalone, Y_train_r_abalone, X_test_r_abalone, Y_test_r_abalone, 0.0001, sigmoidal, 10000, prt = False))

### DIGITS ###
x_digits = dts.X_digits
y_digits = dts.Y_digits
x_digits_r = dts.X_rem_digits
y_digits_r = dts.Y_rem_digits

X_train_digits, Y_train_digits, X_test_digits, Y_test_digits = separate(0.3, x_digits, y_digits, S)
X_train_r_digits, Y_train_r_digits, X_test_r_digits, Y_test_r_digits = separate(0.3, x_digits_r, y_digits_r, S)
print("Digits original:")
print("NBG:", nb_gaussiano(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits))
print("KNN:", knn(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits, 15))
print("ID3:", id3(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits))
print("ANN:", neural_net(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits, 0.001, sigmoidal, 1000, prt = False))
print("Digits rem:")
print("NBG:", nb_gaussiano(X_train_r_digits, Y_train_r_digits, X_test_r_digits, Y_test_r_digits))
print("KNN:", knn(X_train_r_digits, Y_train_r_digits, X_test_r_digits, Y_test_r_digits, 15))
print("ID3:", id3(X_train_r_digits, Y_train_r_digits, X_test_r_digits, Y_test_r_digits))
print("ANN:", neural_net(X_train_r_digits, Y_train_r_digits, X_test_r_digits, Y_test_r_digits, 0.001, sigmoidal, 1000, prt = False))

### CANCER ###
x_cancer = dts.X_cancer
y_cancer = dts.Y_cancer
x_cancer_r = dts.X_rem_cancer
y_cancer_r = dts.Y_rem_cancer

X_train_cancer, Y_train_cancer, X_test_cancer, Y_test_cancer = separate(0.3, x_cancer, y_cancer, S)
X_train_r_cancer, Y_train_r_cancer, X_test_r_cancer, Y_test_r_cancer = separate(0.3, x_cancer_r, y_cancer_r, S)
print("Cancer original:")
print("NBG:", nb_gaussiano(X_train_cancer, Y_train_cancer, X_test_cancer, Y_test_cancer))
print("KNN:", knn(X_train_cancer, Y_train_cancer, X_test_cancer, Y_test_cancer, 15))
print("ID3:", id3(X_train_cancer, Y_train_cancer, X_test_cancer, Y_test_cancer))
print("ANN:", neural_net(X_train_cancer, Y_train_cancer, X_test_cancer, Y_test_cancer, 0.0001, sigmoidal, 6000, prt = False))
print("Cancer rem:")
print("NBG:", nb_gaussiano(X_train_r_cancer, Y_train_r_cancer, X_test_r_cancer, Y_test_r_cancer))
print("KNN:", knn(X_train_r_cancer, Y_train_r_cancer, X_test_r_cancer, Y_test_r_cancer, 15))
print("ID3:", id3(X_train_r_cancer, Y_train_r_cancer, X_test_r_cancer, Y_test_r_cancer))
print("ANN:", neural_net(X_train_r_cancer, Y_train_r_cancer, X_test_r_cancer, Y_test_r_cancer, 0.0001, sigmoidal, 6000, prt = False))

### HUMAN ###
x_human = dts.X_human
y_human = dts.Y_human
x_human_r = dts.X_rem_human
y_human_r = dts.Y_rem_human

X_train_human, Y_train_human, X_test_human, Y_test_human = separate(0.3, x_human, y_human, S)
X_train_r_human, Y_train_r_human, X_test_r_human, Y_test_r_human = separate(0.3, x_human_r, y_human_r, S)
print("Human original:")
print("NBG:", nb_gaussiano(X_train_human, Y_train_human, X_test_human, Y_test_human))
print("KNN:", knn(X_train_human, Y_train_human, X_test_human, Y_test_human, 15))
print("ID3:", id3(X_train_human, Y_train_human, X_test_human, Y_test_human))
print("ANN:", neural_net(X_train_human, Y_train_human, X_test_human, Y_test_human, 0.001, sigmoidal, 2500, prt = False))
print("Human rem:")
print("NBG:", nb_gaussiano(X_train_r_human, Y_train_r_human, X_test_r_human, Y_test_r_human))
print("KNN:", knn(X_train_r_human, Y_train_r_human, X_test_r_human, Y_test_r_human, 15))
print("ID3:", id3(X_train_r_human, Y_train_r_human, X_test_r_human, Y_test_r_human))
print("ANN:", neural_net(X_train_r_human, Y_train_r_human, X_test_r_human, Y_test_r_human, 0.001, sigmoidal, 2500, prt = False))

### IRIS ###
x_iris = dts.X_iris
y_iris = dts.Y_iris
x_iris_r = dts.X_rem_iris
y_iris_r = dts.Y_rem_iris

X_train_iris, Y_train_iris, X_test_iris, Y_test_iris = separate(0.3, x_iris, y_iris, S)
X_train_r_iris, Y_train_r_iris, X_test_r_iris, Y_test_r_iris = separate(0.3, x_iris_r, y_iris_r, S)
print("Iris original:")
print("NBG:", nb_gaussiano(X_train_iris, Y_train_iris, X_test_iris, Y_test_iris))
print("KNN:", knn(X_train_iris, Y_train_iris, X_test_iris, Y_test_iris, 15))
print("ID3:", id3(X_train_iris, Y_train_iris, X_test_iris, Y_test_iris))
print("ANN:", neural_net(X_train_iris, Y_train_iris, X_test_iris, Y_test_iris, 0.01, sigmoidal, 15000, prt = False))
print("Iris rem:")
print("NBG:", nb_gaussiano(X_train_r_iris, Y_train_r_iris, X_test_r_iris, Y_test_r_iris))
print("KNN:", knn(X_train_r_iris, Y_train_r_iris, X_test_r_iris, Y_test_r_iris, 15))
print("ID3:", id3(X_train_r_iris, Y_train_r_iris, X_test_r_iris, Y_test_r_iris))
print("ANN:", neural_net(X_train_r_iris, Y_train_r_iris, X_test_r_iris, Y_test_r_iris, 0.01, sigmoidal, 15000, prt = False))

### WINE ###
x_wine = dts.X_wine
y_wine = dts.Y_wine
x_wine_r = dts.X_rem_wine
y_wine_r = dts.Y_rem_wine

X_train_wine, Y_train_wine, X_test_wine, Y_test_wine = separate(0.3, x_wine, y_wine, S)
X_train_r_wine, Y_train_r_wine, X_test_r_wine, Y_test_r_wine = separate(0.3, x_wine_r, y_wine_r, S)
print("Wine original:")
print("NBG:", nb_gaussiano(X_train_wine, Y_train_wine, X_test_wine, Y_test_wine))
print("KNN:", knn(X_train_wine, Y_train_wine, X_test_wine, Y_test_wine, 15))
print("ID3:", id3(X_train_wine, Y_train_wine, X_test_wine, Y_test_wine))
print("ANN:", neural_net(X_train_wine, Y_train_wine, X_test_wine, Y_test_wine, 0.004, tangenteh, 6000, prt = False))
print("Wine rem:")
print("NBG:", nb_gaussiano(X_train_r_wine, Y_train_r_wine, X_test_r_wine, Y_test_r_wine))
print("KNN:", knn(X_train_r_wine, Y_train_r_wine, X_test_r_wine, Y_test_r_wine, 15))
print("ID3:", id3(X_train_r_wine, Y_train_r_wine, X_test_r_wine, Y_test_r_wine))
print("ANN:", neural_net(X_train_r_wine, Y_train_r_wine, X_test_r_wine, Y_test_r_wine, 0.004, tangenteh, 6000, prt = False))