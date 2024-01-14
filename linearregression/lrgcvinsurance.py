#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:33:30 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Medical Cost Personal")
df = pd.read_csv("insurance.csv")
dummy_df = pd.get_dummies(df,drop_first=True)
x = dummy_df.drop('charges',axis=1)
y = dummy_df['charges']

lr = LinearRegression()
lr.fit(x,y)

print(lr.coef_)
print(lr.intercept_)

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
result = cross_val_score(lr, x,y,cv=kfold,scoring='r2')

print(result)
print(result.mean())

###########KNN Regression###############

scaler = StandardScaler()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
knn = KNeighborsRegressor()
pipe = Pipeline([('scaline',scaler),('knn_model',knn)])
params = {'knn_model__n_neighbors':np.arange(1,31)}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold,scoring='r2')
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)