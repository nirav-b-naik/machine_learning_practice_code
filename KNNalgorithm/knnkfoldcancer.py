#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:40:03 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin")
df = pd.read_csv("BreastCancer.csv")
df=df.drop('Code',axis=1)
dum_df=pd.get_dummies(df,drop_first=True)

x=dum_df.drop('Class_Malignant',axis=1)
y=dum_df['Class_Malignant']

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2022)

roc=list()

for i in range(1,30,2):    
    knn=KNeighborsClassifier(n_neighbors=i)
    results=cross_val_score(knn,x,y,
                            scoring='roc_auc',
                            cv=kfold)
    
    #print(f'{i}: ',results,end='\t')
    #print(round(results.mean(),6))
    roc.append(round(results.mean(),6))

print(f'max: {np.max(roc)}, k: {2*np.argmax(roc)+1}')
    
###########################GridsearchCV###################

from sklearn.model_selection import GridSearchCV
df = pd.read_csv("BreastCancer.csv")
df=df.drop('Code',axis=1)
dum_df=pd.get_dummies(df,drop_first=True)

x=dum_df.drop('Class_Malignant',axis=1)
y=dum_df['Class_Malignant']
kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2022)
knn=KNeighborsClassifier()

params={'n_neighbors':np.arange(1,30,1)}

gcv=GridSearchCV(knn, param_grid=params, scoring='roc_auc',cv=kfold)
gcv.fit(x,y) 
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv=pd.DataFrame(gcv.cv_results_)
