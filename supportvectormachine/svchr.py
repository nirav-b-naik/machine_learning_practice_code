#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:57:25 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics/HR_comma_sep.csv")
dum_df= pd.get_dummies(df,drop_first=True)
x = dum_df.drop('left',axis=1)
y = dum_df["left"]

###########Linear

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm = SVC(probability=True,kernel='linear')

params = {'C':np.linspace(0.001,5,10)}
gcv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc')
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

###########rbf

svm = SVC(probability=True,kernel='rbf')

params = {'C':np.linspace(0.001,5,5),
          'gamma':np.linspace(0.001,5,5)}
gcv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc')
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

###########polynomial

svm = SVC(probability=True,kernel='poly')

params = {'C':np.linspace(0.001,5,5),
          'gamma':np.linspace(0.001,5,5),
          'degree': [2,3,4]}
gcv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc')
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

