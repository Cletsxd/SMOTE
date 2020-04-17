# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:50:31 2020

@author: 4PF41LA_RS6
"""

from data import Datasets
import numpy as np
import random
import math

def get_minority(name_class, Y, X):
    x = []
    
    for y in range(len(Y)):
        if Y[y][0] == name_class:
            x.append(X[y])
    
    x = np.asarray(x)
    
    return x

def distance(s, sam, atts):
    sq = 0
    
    for a in range(atts):
        sq = sq + ((s[a] - sam[a]) ** 2)
    
    return math.sqrt(sq)

def knearest(s, Sample, k, atts):
    dist = []
    for sam in Sample:
        d = distance(s, sam, atts)
        
        if d != 0.0:
            dist.append(distance(s, sam, atts))
    
    dist = np.asarray(dist)    
    r = np.argpartition(dist, k)

    return r[:k]

def smote(T, N, k, Sample):
    print("T:", T, "N:", N, "k:", k)
    
    if N < 100:
        print(Sample)
    
    numattrs = Sample.shape[1]
    newindex = 0
    Synthetic = []
    
    for i in range(int(T)):
        print(i)
        nnarray = knearest(Sample[i], Sample, k, numattrs)
        print(i, ".", 1)
        
        ## Populate
        NN = N/100
        
        while NN != 0:
            nn = random.randint(0, k - 1)
            at = []
            
            for attr in range(0, numattrs):
                dif = Sample[nnarray[nn]][attr] - Sample[i][attr]
                gap = random.uniform(0, 1)
                
                at.append(Sample[i][attr] + gap * dif)
            
            print(at)
            Synthetic.append(at)
            newindex += 1
            NN = NN - 1
    
    return Synthetic

dts = Datasets()

#dts.remove_data(10)
dts.remove_data(1)

x_iris = dts.X_abalone
y_iris = dts.Y_abalone
x_r_iris = dts.X_rem_abalone
y_r_iris = dts.Y_rem_abalone

# print(dts.data_info(y_iris), dts.data_info(y_r_iris))

def smote_data(y_r):
    ### N compute ###
    count_y, ind_mayor, class_mayor = dts.data_info(y_r)
    
    name_class = list(count_y[0])
    count_class = list(count_y[1])
    
    print(name_class, count_class, count_class[ind_mayor[0][0]])
    
    N_class = []
    T_class = []
    T_name_class = []
    N_mayor = count_class[ind_mayor[0][0]]
    
    for c in range(len(count_class)):
        if count_class[c] != N_mayor:
            N_class.append(count_class[c])
            T_class.append(count_class[c])
            T_name_class.append(name_class[c])
    
    for n in range(len(N_class)):
        N_class[n] = (int((N_mayor - N_class[n]) / N_class[n])) * 100
    
    Syns = []
    for t in range(len(T_class)):
        T_minority_class_samples = get_minority(T_name_class[t], y_r_iris, x_r_iris)
        Syns.append(smote(T_class[t], N_class[t], 5, T_minority_class_samples))