#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:57:02 2022

@author: dai
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
os.chdir(".")
df = pd.read_csv("insurance.csv")

dum_insur = pd.get_dummies(df,drop_first=True)
x = dum_insur.drop('charges',axis=1)
y = dum_insur['charges']
scaler = MinMaxScaler()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
knn = KNeighborsRegressor()
pipe = Pipeline([('scaline',scaler),('knn_model',knn)])
params = {'knn_model__n_neighbors':np.arange(1,31)}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold,scoring='r2')
gcv.fit(x, y)
print(gcv.best_params_)
print(gcv.best_score_)
best_est=gcv.best_estimator_

##############################from other file ##########

tst_insure=pd.read_csv('tst_insure.csv')
dum_tst_insur = pd.get_dummies(tst_insure,drop_first=True)
predictions=best_est.predict(dum_tst_insur)

predictions=gcv.predict(dum_tst_insur)
