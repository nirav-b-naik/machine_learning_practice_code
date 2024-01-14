#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:46:21 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Glass Identification/Glass.csv")
x = df.drop("Type",axis=1)
y = df["Type"]

kfold =  StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

le = LabelEncoder()
le_y = le.fit_transform(y)

#################### Linear with Minmax

sclmm=MinMaxScaler()
svm = SVC(probability=True,kernel='linear',random_state=2022)

pipe1=Pipeline([('sclmm',sclmm),('smv_model',svm)])

params = {'smv_model__C':np.linspace(0.001,5,10),
          'smv_model__decision_function_shape' : ['ovo','ovr']}
gcv =  GridSearchCV(pipe1, param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=1)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

#################### rbf with Minmax

svm = SVC(probability=True,kernel='rbf',random_state=2022)

pipe2=Pipeline([('sclmm',sclmm),('smv_model',svm)])

params = {'smv_model__C':np.linspace(0.001,5,10),
          'smv_model__gamma': np.linspace(0.001,5,10),
          'smv_model__decision_function_shape' : ['ovo','ovr']}
gcv =  GridSearchCV(pipe2, param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=1)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

#################### Linear with standardscaler

sclstd=StandardScaler()
svm = SVC(probability=True,kernel='linear',random_state=2022)

pipe3=Pipeline([('sclstd',sclstd),('smv_model',svm)])

params = {'smv_model__C':np.linspace(0.001,5,10),
          'smv_model__decision_function_shape' : ['ovo','ovr']}
gcv =  GridSearchCV(pipe3, param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=1)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

#################### rbf with standardscaler

svm = SVC(probability=True,kernel='rbf',random_state=2022)

pipe4=Pipeline([('sclstd',sclstd),('smv_model',svm)])

params = {'smv_model__C':np.linspace(0.001,5,10),
          'smv_model__gamma': np.linspace(0.001,5,10),
          'smv_model__decision_function_shape' : ['ovo','ovr']}
gcv =  GridSearchCV(pipe4, param_grid=params,scoring='neg_log_loss',cv=kfold,verbose=1)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

################################output
'''
#################### Linear with Minmax

Fitting 5 folds for each of 20 candidates, totalling 100 fits
{'smv_model__C': 5.0, 'smv_model__decision_function_shape': 'ovo'}
-0.9968060981384189

#################### rbf with Minmax

Fitting 5 folds for each of 200 candidates, totalling 1000 fits
{'smv_model__C': 5.0, 'smv_model__decision_function_shape': 'ovo', 'smv_model__gamma': 5.0}
-0.8209377543154949

#################### Linear with standardscaler

Fitting 5 folds for each of 20 candidates, totalling 100 fits
{'smv_model__C': 2.2227777777777775, 'smv_model__decision_function_shape': 'ovo'}
-0.9753914353701294

#################### rbf with standardscaler

Fitting 5 folds for each of 200 candidates, totalling 1000 fits
{'smv_model__C': 1.1118888888888887, 'smv_model__decision_function_shape': 'ovo', 'smv_model__gamma': 0.5564444444444444}
-0.8186921737245922
'''