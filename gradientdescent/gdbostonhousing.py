#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:44:41 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import SGDRegressor

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing')

df= pd.read_csv('Boston.csv')

x=df.drop('medv',axis=1)
y=df['medv']

scaler=MinMaxScaler()

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

sgd=SGDRegressor(random_state=2022)

#######################grid search without scaling

pipe1=Pipeline([('sgd_model',sgd)])

params={'sgd_model__eta0':np.linspace(0.001,0.7,10),'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive']}

gcv=GridSearchCV(pipe1, param_grid=params,scoring='r2',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

#######################grid search with Minmaxscaling

scaler=MinMaxScaler()

pipe2=Pipeline([('scl_std',scaler),('sgd_model',sgd)])

params={'sgd_model__eta0':np.linspace(0.001,0.7,10),'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive']}

gcv=GridSearchCV(pipe2, param_grid=params,scoring='r2',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

#######################grid search with Standardscaling

scaler=StandardScaler()

pipe3=Pipeline([('scl_std',scaler),('sgd_model',sgd)])

params={'sgd_model__eta0':np.linspace(0.001,0.7,10),'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive']}

gcv=GridSearchCV(pipe3, param_grid=params,scoring='r2',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
