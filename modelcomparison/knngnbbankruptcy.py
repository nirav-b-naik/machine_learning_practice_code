#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:46:53 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Bankruptcy")
df = pd.read_csv("Bankruptcy.csv")

x = df.drop(['NO', 'D', 'YR'],axis=1)
y = df['D']

scaler=StandardScaler()

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

############KNN############################################################

knn=KNeighborsClassifier()

pipe=Pipeline([('scl_std',scaler),('knn_model',knn)])

params={'knn_model__n_neighbors': np.arange(1,30)}

gcv=GridSearchCV(pipe, param_grid=params, scoring='roc_auc',cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv=pd.DataFrame(gcv.cv_results_)

best_est=gcv.best_estimator_

############################GaussianNB####################################

gnb = GaussianNB()

pipe=Pipeline([('scl_std',scaler),('gnb_model',gnb)])

result = cross_val_score(pipe, x,y,scoring='roc_auc',cv=kfold)
print(result)
print(result.mean())

'''print(pipe.get_params())

params={'gnb_model__var_smoothing': }

gcv=GridSearchCV(pipe, param_grid=params, scoring='roc_auc',cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv1=pd.DataFrame(gcv.cv_results_)
'''

##################test file read#########################################

test_bnk=pd.read_csv("testBankruptcy.csv")
test_bnk = test_bnk.drop(['NO'],axis=1)
predictions=best_est.predict(test_bnk)

