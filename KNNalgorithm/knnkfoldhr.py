#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:01:04 2022

@author: dai
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics")
hr_df =pd.read_csv("HR_comma_sep.csv")

df_dummy = pd.get_dummies(hr_df,drop_first=True)
x = df_dummy.drop('left',axis=1)
y = df_dummy['left']

ll = []

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
for i in range(1,31,2):
    knn = KNeighborsClassifier(n_neighbors=i)
    results = cross_val_score(knn, x,y,scoring='roc_auc',cv = kfold)
    ll.append(results.mean())

print(f'max: {np.max(ll)}, k: {2*np.argmax(ll)+1}')


#######################GridSearchCV#######################
from sklearn.model_selection import GridSearchCV
hr_df = pd.read_csv("HR_comma_sep.csv")
dum_df = pd.get_dummies(hr_df,drop_first=True)

x = dum_df.drop('left',axis=1)
y = dum_df['left']

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
knn = KNeighborsClassifier()
params = {'n_neighbors':np.arange(1,31,2)}

gcv = GridSearchCV(knn, param_grid=params,scoring='roc_auc',cv=kfold)
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
pd_gcv = pd.DataFrame(gcv.cv_results_)
