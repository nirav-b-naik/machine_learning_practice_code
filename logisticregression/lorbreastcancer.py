#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:08:08 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

cancer = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin/BreastCancer.csv",index_col=0)
dum_cancer = pd.get_dummies(cancer,drop_first=True)
x = dum_cancer.drop('Class_Malignant',axis=1)
y = dum_cancer["Class_Malignant"]
xtrain,xtest,ytrain,ytest = train_test_split(x,
                                             y,
                                             test_size=0.3,
                                             stratify=y,
                                             random_state=2022)

logreg = LogisticRegression()
logreg.fit(xtrain,ytrain)

y_pred_prob = logreg.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob))

###################### KFold CV ############################################

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
logreg = LogisticRegression()
result = cross_val_score(logreg,x,y,scoring='roc_auc',cv=kfold)

print(result.mean())