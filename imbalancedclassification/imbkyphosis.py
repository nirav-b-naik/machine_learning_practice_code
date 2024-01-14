#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:40:20 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
import graphviz

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Kyphosis')

df=pd.read_csv('Kyphosis.csv')

dum_df=pd.get_dummies(df,drop_first=True)

x=dum_df.drop('Kyphosis_present',axis=1)

y=dum_df['Kyphosis_present']

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,
                                                 stratify=y,
                                                 random_state=2022)

lr = LogisticRegression()
################### w/o Balancing ######################
lr.fit(xtrain, ytrain)
y_pred = lr.predict(xtest)
print(classification_report(ytest,y_pred))

################## Randomvoersampling (naive)

ros=RandomOverSampler(random_state=2022)

xreshape,yreshape=ros.fit_resample(x, y)

svm=SVC(probability=True, kernel='linear',random_state=2022)

params={'C':np.linspace(0.001,5,10)}

gcv=GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc')

gcv.fit(xreshape,yreshape)

print(gcv.best_params_)
print(gcv.best_score_)

################## oversampling SMOTE

smote=SMOTE(random_state=2022)

xreshape,yreshape=smote.fit_resample(x, y)

svm=SVC(probability=True, kernel='linear',random_state=2022)

params={'C':np.linspace(0.001,5,10)}

gcv=GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc')

gcv.fit(xreshape,yreshape)

print(gcv.best_params_)
print(gcv.best_score_)

################## oversampling ADASYN

adasyn=ADASYN(random_state=2022)

xreshape,yreshape=adasyn.fit_resample(x, y)

svm=SVC(probability=True, kernel='linear',random_state=2022)

params={'C':np.linspace(0.001,5,10)}

gcv=GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc')

gcv.fit(xreshape,yreshape)

print(gcv.best_params_)
print(gcv.best_score_)
