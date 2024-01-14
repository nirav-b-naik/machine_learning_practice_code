#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 10:39:21 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
#from sklearn.

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Bankruptcy/Bankruptcy.csv",index_col=0)
x = df.drop(['D', 'YR'],axis=1)
y = df['D']

###########RBF

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm = SVC(probability=True,kernel='rbf',random_state=2022)

params = {'C':np.linspace(0.001,5,10),
          'gamma':np.linspace(0.001,5,10)}
gcv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc',verbose=1)
gcv.fit(x,y)

print(gcv.best_params_) 
# {'C': 5.0, 'gamma': 0.001}

print(gcv.best_score_)
# 0.797548605240913

###########RBF with bagging

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm = SVC(probability=True,kernel='rbf',random_state=2022)

bag=BaggingClassifier(base_estimator=svm,random_state=2022)

params = {'base_estimator__C':np.linspace(0.001,5,10),
          'base_estimator__gamma':np.linspace(0.001,5,10),
          'n_estimators':[10,50,100]}
gcv = GridSearchCV(bag, param_grid=params,cv=kfold,scoring='roc_auc',verbose=1)
gcv.fit(x,y)

print(gcv.best_params_)

print(gcv.best_score_)

##### 2 params only
#{'base_estimator__C': 4.444555555555556, 'base_estimator__gamma': 0.5564444444444444}
#0.7879966187658496

##### 3 params only
#{'base_estimator__C': 5.0, 'base_estimator__gamma': 0.001, 'n_estimators': 50}
#0.800422654268808

