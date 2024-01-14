#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:01:39 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

########### concrete strength LinearRegression ################

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']

lr=LinearRegression()
scaler= MinMaxScaler()
poly=PolynomialFeatures()

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

pipe=Pipeline([('poly',poly),('scl_mm',scaler),('lr_model',lr)])

params={'poly__degree': [1,2,3,4]}

gcv=GridSearchCV(pipe, param_grid=params,scoring='r2',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

########### concrete strength SGDRegressor ################

sgd=SGDRegressor()
scaler= MinMaxScaler()
poly=PolynomialFeatures()

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

pipe=Pipeline([('poly',poly),('scl_mm',scaler),('sgd_model',sgd)])

params={'sgd_model__eta0':np.linspace(0.001,0.7,10),
        'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive'],
        'poly__degree': [1,2,3,4]}

gcv=GridSearchCV(pipe, param_grid=params,scoring='r2',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
'''
{'poly__degree': 4, 'sgd_model__eta0': 0.001, 'sgd_model__learning_rate': 'adaptive'}
0.8245276627005926
'''