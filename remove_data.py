# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:54:40 2020

@author: 4PF41LA_RS6
"""
from data import Datasets
import numpy as np

dts = Datasets()

#dts.remove_data(10)
dts.remove_data(1)

y_abalone = dts.Y_abalone
y_abalone_r = dts.Y_rem_abalone
print("Abalone:")
print(dts.data_info(y_abalone), dts.data_info(y_abalone_r), dts.reduce)

y_digits = dts.Y_digits
y_digits_r = dts.Y_rem_digits
print("\nDigits:")
print(dts.data_info(y_digits), dts.data_info(y_digits_r), dts.reduce)

y_cancer = dts.Y_cancer
y_cancer_r = dts.Y_rem_cancer
print("\nCancer:")
print(dts.data_info(y_cancer), dts.data_info(y_cancer_r), dts.reduce)

y_human = dts.Y_human
y_human_r = dts.Y_rem_human
print("\nHuman:")
print(dts.data_info(y_human), dts.data_info(y_human_r), dts.reduce)

y_iris = dts.Y_iris
y_iris_r = dts.Y_rem_iris
print("\nIris:")
print(dts.data_info(y_iris), dts.data_info(y_iris_r), dts.reduce)

y_wine = dts.Y_wine
y_wine_r = dts.Y_rem_wine
print("\nWine:")
print(dts.data_info(y_wine), dts.data_info(y_wine_r), dts.reduce)