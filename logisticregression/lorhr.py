#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:45:28 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import GridSearchCV

hr=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics/HR_comma_sep.csv')
dum_hr=pd.get_dummies(hr,drop_first=True)
x=dum_hr.drop('left',axis=1)
y=dum_hr['left']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,random_state=2022)

logreg=LogisticRegression(max_iter=100)
logreg.fit(xtrain,ytrain)
y_pred_prob=logreg.predict_proba(xtest)[:,0]

#AUC_ROC_SCORE_RAISE
print(roc_auc_score(ytest, y_pred_prob))



###################### KFold CV ############################################

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
logreg = LogisticRegression(max_iter=1000)
result = cross_val_score(logreg,x,y,scoring='roc_auc',cv=kfold)

print(result.mean())

###################### GridSearchCV ############################################

params={'penalty': ['l1','l2','elasticnet','none'],'solver':['newton-cg','lbfgs','liblinear','sag','saga']}

gcv=GridSearchCV(logreg, param_grid=params,scoring='roc_auc',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

gcv_df=pd.DataFrame(gcv.cv_results_)
