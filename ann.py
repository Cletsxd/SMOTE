# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:05:23 2020

@author: 4PF41LA_RS6
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time
import csv

from IPython.display import clear_output
from sklearn.datasets import make_circles

# CLASE PARA LAS CAPAS DE UNA ANN
class NeuralLayer():

    # n_conn: número de conexiones de las neuronas
    # n_neur: número de neuronas
    # act_f: función de activación
    def __init__(self, n_conn, n_neur, act_f):

        self.act_f=act_f

        # vector b: Bias, tantos como neuronas, es decir b=n_neur
        # random values: (-1, 1)
        self.b = np.random.rand(1, n_neur) * 2 -1

        # matriz w [n_conn x n_neur]: pesos, ej: 3 conexiones => 1 neurona
        self.w = np.random.rand(n_conn, n_neur) * 2 -1

# Crear TOPOLOGÍA DE LA ANN
def create_nn(topology, act_f):
    
    # CREACIÓN DE LA ANN, VECTOR DE CAPAS
    neural_net = []

    for l, layer in enumerate(topology[: -1]):
        # Inserta las capas a la ANN
        # índice l: número de capas
        neural_net.append(NeuralLayer(topology[l], topology[l+1], act_f))

    return neural_net

# FUNCIÓN DE ENTRENAMIENTO
# X: datos entrada
# Y: datos salida esperada
def train(neural_net, X, Y, e2medio, learning_rate=0.5, train=True):
    # X, Y: matrices
    
    # vector output: salida
    output = [(None, X)]

    # Forward pass
    for l, layer in enumerate(neural_net):
        # l: recorre las capas de la neural_net
        # z: suma ponderada
        z = output[-1][1] @ neural_net[l].w + neural_net[l].b

        # a: salida capa1
        a = neural_net[l].act_f[0](z)

        output.append((z, a))

    if train:
        # Backward pass
        # Backpropagation algorithm

        deltas = []
        # len(neural_net): número de capas de la ANN
        for l in reversed(range(0, len(neural_net))):
            # output[l+1][0]: suma ponderada
            z = output[l+1][0]
            # output[l+1][1]: activación
            a = output[l+1][1]
            if l == len(neural_net) -1:
                # Calcular delta última capa con respecto al coste
                deltas.insert(0, e2medio[1](a, Y) * neural_net[l].act_f[1](a))
            else:
                # Calcular capa respecto de capa previa
                deltas.insert(0, deltas[0] @ _w.T * neural_net[l].act_f[1](a))

            _w = neural_net[l].w

            # Gradiente descendiente
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * learning_rate
            neural_net[l].w = neural_net[l].w - output[l][1].T @ deltas[0] * learning_rate

    return output[-1][1]

# Función de activación
# SIGMOIDAL
# sigmodial[0](x) // función de activación Sigmoidal
# sigmodial[1](x) // derivada de la Sigmoidal
sigmoidal = (lambda x: 1 / (1 + np.exp(-x)),
            lambda x: x * (1 - x))

tangenteh = (lambda x: np.tanh(x),
            lambda x: 1 - (np.tanh(x) ** 2))

# Función de costo
# ERROR CUADRÁTICO MEDIO
# yp: salida real de la ANN
# yr: salida predicha de la entrada
# e2medio[0](yp, yr) // función
# e2medio[1](yp, yr) // derivada
e2medio = (lambda yp, yr: np.mean((yp, yr)) ** 2,
            lambda yp, yr: (yp - yr))

# Función de costo
# ERROR CUADRÁTICO MEDIO
# yp: salida real de la ANN
# yr: salida predicha de la entrada
# e2medio[0](yp, yr) // función
# e2medio[1](yp, yr) // derivada
e2medio = (lambda yp, yr: np.mean((yp, yr)) ** 2,
            lambda yp, yr: (yp - yr))

def fit(X_train, Y_train, X_test, Y_test, topology, lr, act_foo, epochs, prt):
    neural_net = create_nn(topology, act_foo)
    
    last = topology[-1]
    
    Y_e = []
    Y_t = []

    for y in Y_train:
        O = np.zeros(last)
        O[int(y)] = 1
        
        Y_e.append(O)
    
    for y in Y_test:
        O = np.zeros(last)
        O[int(y)] = 1
        
        Y_t.append(O)
    
    Y_e = np.asarray(Y_e)
    Y_t = np.asarray(Y_t)
    loss = []
    loss_t = []
    
    for i in range(epochs):
        out = train(neural_net, X_train, Y_e, e2medio, learning_rate=lr)
        loss.append(e2medio[0](out, Y_e))
        
        out = train(neural_net, X_test, Y_t, e2medio, learning_rate=lr, train = False)
        loss_t.append(e2medio[0](out, Y_t))
        
        if i % 25 == 0 and prt:
            #print(out)
        
            print("e:", loss[-1])
            print("e_t:", loss_t[-1])
    
    outt = train(neural_net, X_test, Y_t, e2medio, learning_rate=lr, train = False)
    
    Y_pred = []
    for o in outt:
        Y_pred.append(np.where(max(o) == o)[0][0])

    Y_pred = np.asarray(Y_pred)
    
    return Y_pred

def accuracy_score(real, pred):
    count = 0
    
    for r in range(len(real)):
        if real[r] == pred[r]:
            count += 1
    
    return count/len(real)