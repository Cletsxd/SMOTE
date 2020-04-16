# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:38:57 2020

@author: 4PF41LA_RS6
"""

from sklearn import datasets
import numpy as np
import csv
import random

class Datasets:
    def __init__(self):
        X_abalone = []
        Y_abalone = []
        with open('abalone.csv', newline='') as File:
                reader = csv.reader(File)
                for r in reader:
                    X_abalone.append([float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7])])
                    
                    if r[8] == 'M':
                        Y_abalone.append(0.0)
                    else:
                        Y_abalone.append(1.0)
                        
        X_human = []
        Y_human = []
        with open('human.csv', newline='') as File:
                reader = csv.reader(File)
                for r in reader:
                    X_human.append([float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7])])
        
                    if r[10] == "WALKING":
                        Y_human.append(0.0)
                    elif r[10] == "WALKING_UPSTAIRS":
                        Y_human.append(1.0)
                    elif r[10] == "WALKING_DOWNSTAIRS":
                        Y_human.append(2.0)
                    elif r[10] == "SITTING":
                        Y_human.append(3.0)
                    elif r[10] == "STANDING":
                        Y_human.append(4.0)
                    elif r[10] == "LAYING":
                        Y_human.append(5.0)
        
        self.X_abalone = np.asarray(X_abalone)
        Y_abalone = np.asarray(Y_abalone)
        self.Y_abalone = np.reshape(Y_abalone, (4177, 1))
        
        cancer = datasets.load_breast_cancer()
        self.X_cancer = cancer.data
        Y_cancer = cancer.target
        self.Y_cancer = np.reshape(Y_cancer, (569, 1))
        
        digits = datasets.load_digits()
        self.X_digits = digits.data
        Y_digits = digits.target
        self.Y_digits = np.reshape(Y_digits, (1797, 1))
        
        self.X_human = np.asarray(X_human)
        Y_human = np.asarray(Y_human)
        self.Y_human = np.reshape(Y_human, (7352, 1))
        
        iris = datasets.load_iris()
        self.X_iris = iris.data
        Y_iris = iris.target
        self.Y_iris = np.reshape(Y_iris, (150, 1))
        
        wine = datasets.load_wine()
        self.X_wine = wine.data
        Y_wine = wine.target
        self.Y_wine = np.reshape(Y_wine, (178, 1))

    def data_info(self, Y):
        count_y = np.asarray(np.unique(Y, return_counts = True))
        ind_mayor = np.where(max(count_y[1]) == count_y[1])
        class_mayor = count_y[0][ind_mayor]
        
        return count_y, ind_mayor, class_mayor
    
    def remove_a_data(self, S, X, Y):
        random.seed(S)        
        count_y = np.asarray(np.unique(Y, return_counts = True))
        ind_mayor = np.where(max(count_y[1]) == count_y[1])
        class_mayor = count_y[0][ind_mayor]

        count_reduce = int(self.reduce * Y.shape[0])
        
        rem = []
        
        for _ in range(count_reduce):
            shoise = random.randint(0, Y.shape[0] - 1)
            
            while Y[shoise][0] == class_mayor[0]:
                shoise = random.randint(0, Y.shape[0] - 1)
            
            rem.append(shoise)
        
        rem = np.asarray(rem)
        rem = np.unique(rem)
        
        X_new = []
        Y_new = []
        for i in range(Y.shape[0]):
            if not i in rem:
                X_new.append(X[i])
                Y_new.append(Y[i])
        
        X_new = np.asarray(X_new)
        Y_new = np.asarray(Y_new)
        
        return X_new, Y_new
    
    def remove_data(self, S):
        random.seed(S)
        self.reduce = random.choice([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
        
        self.X_rem_abalone, self.Y_rem_abalone = self.remove_a_data(S, self.X_abalone, self.Y_abalone)
        self.X_rem_digits, self.Y_rem_digits = self.remove_a_data(S, self.X_digits, self.Y_digits)
        self.X_rem_cancer, self.Y_rem_cancer = self.remove_a_data(S, self.X_cancer, self.Y_cancer)
        self.X_rem_human, self.Y_rem_human = self.remove_a_data(S, self.X_human, self.Y_human)
        self.X_rem_iris, self.Y_rem_iris = self.remove_a_data(S, self.X_iris, self.Y_iris)
        self.X_rem_wine, self.Y_rem_wine = self.remove_a_data(S, self.X_wine, self.Y_wine)