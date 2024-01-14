#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:12:47 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Bankruptcy/Bankruptcy.csv",index_col=0)
x = df.drop(['D', 'YR'],axis=1)
y = df['D']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,stratify=y,random_state=2022)

lda= LinearDiscriminantAnalysis()
svm_l= SVC(kernel='linear',probability=True,random_state=2022)
svm_c= SVC(kernel='rbf',probability=True,random_state=2022)
rfc= RandomForestClassifier(random_state=2022)
logreg=LogisticRegression(random_state=2022)
xgb=XGBClassifier(random_state=2022)
dtc= DecisionTreeClassifier(random_state=2022)

#################### with final_estimator=RandomForestClassifier

models=[('logreg',logreg),('svm_l',svm_l),('svm_c',svm_c),('lda',lda),('dtc',dtc)]

stack=StackingClassifier(estimators=models,final_estimator=rfc,verbose=10,passthrough=False)

stack.fit(xtrain,ytrain)

y_pred=stack.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred))
# 0.91875 - with passthrough
# 0.84125 - without passthrough

#################### with final_estimator=XGBClassifier

models=[('logreg',logreg),('svm_l',svm_l),('svm_c',svm_c),('lda',lda),('dtc',dtc)]

stack=StackingClassifier(estimators=models,final_estimator=xgb,verbose=10,passthrough=False)

stack.fit(xtrain,ytrain)

y_pred=stack.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred))
# 0.8725 - with passthrough
# 0.8625 - without passthrough

#################### with final_estimator=RandomForestClassifier with gridsearchcv

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2022)

models=[('logreg',logreg),('svm_l',svm_l),('svm_c',svm_c),('lda',lda),('dtc',dtc)]

stack=StackingClassifier(estimators=models,final_estimator=rfc,verbose=10,passthrough=True)

params={'dtc__max_depth': [None,3],
        'svm_l__C':[1,0.5],
        'svm_c__C': [1,0.5]}

gcv=GridSearchCV(stack, param_grid=params,scoring='roc_auc',cv=kfold,verbose=10)

gcv.fit(x,y)

print(gcv.best_params_)
# {'dtc__max_depth': 3, 'svm_c__C': 0.5, 'svm_l__C': 0.5} - without passthrough
# {'dtc__max_depth': 3, 'svm_c__C': 0.5, 'svm_l__C': 1} - with passthrough

print(gcv.best_score_)
# 0.8723161453930685 - without passthrough
# 0.9139475908706677 - with passthrough
