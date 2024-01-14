#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:19:30 2022

@author: dai
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Satellite Imaging/Satellite.csv",sep=';')
x = df.drop("classes",axis=1)
y = df["classes"]

kfold =  StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

le = LabelEncoder()
le_y = le.fit_transform(y)

#################### Linear

svm = SVC(probability=True,kernel='linear')
params = {'C':np.linspace(0.001,5,5),
          'decision_function_shape' : ['ovo','ovr']}
gcv =  GridSearchCV(svm, param_grid=params,scoring='neg_log_loss',cv=kfold)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

#################### rbf

svm = SVC(probability=True,kernel='rbf')
params = {'C':np.linspace(0.001,5,5),
          'gamma':np.linspace(0.001,5,5),
          'decision_function_shape' : ['ovo','ovr']}
gcv =  GridSearchCV(svm, param_grid=params,scoring='neg_log_loss',cv=kfold)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)