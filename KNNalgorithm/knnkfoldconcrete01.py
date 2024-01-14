#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:35:10 2022

@author: dai
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

########### concrete strength ################

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('scl_std',scaler),('knn_model',knn)])
print(pipe.get_params())
params={'knn_model__n_neighbors':np.arange(1,31)}
gcv = GridSearchCV(pipe, param_grid=params,scoring='r2',cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv = pd.DataFrame(gcv.cv_results_)
