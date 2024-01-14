#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:55:51 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline

cancer = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin/BreastCancer.csv",index_col=0)
dum_cancer = pd.get_dummies(cancer,drop_first=True)
x = dum_cancer.drop('Class_Malignant',axis=1)
y = dum_cancer["Class_Malignant"]
xtrain,xtest,ytrain,ytest = train_test_split(x,
                                             y,
                                             test_size=0.3,
                                             stratify=y,
                                             random_state=2022)

da=LinearDiscriminantAnalysis()
da.fit(xtrain,ytrain)

y_pred_prob = da.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob))

###################### LinearDiscriminantAnalysis ############################################

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
da=LinearDiscriminantAnalysis()
result = cross_val_score(da,x,y,scoring='roc_auc',cv=kfold,verbose=1)

print(result.mean())

###################### QuadraticDiscriminantAnalysis ############################################

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
qda=QuadraticDiscriminantAnalysis()
result = cross_val_score(qda,x,y,scoring='roc_auc',cv=kfold,verbose=1)

print(result.mean())