#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:31:42 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Kyphosis/Kyphosis.csv")
df_dummy = pd.get_dummies(df,drop_first=True)

x = df_dummy.drop("Kyphosis_present",axis=1)
y = df_dummy["Kyphosis_present"]

#kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
sgd =SGDClassifier(loss='log_loss')

poly = PolynomialFeatures()

pipe = Pipeline([('poly',poly),('sgd_model',sgd)])

params = {'poly__degree':[1,2,3,4],'sgd_model__eta0':np.linspace(0.001,0.7,10),'sgd_model__learning_rate': ['constant','optimal','invscaling','adaptive']}

gcv = GridSearchCV(pipe, param_grid=params,scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_estimator_)
print(gcv.best_params_)
print(gcv.best_score_)
