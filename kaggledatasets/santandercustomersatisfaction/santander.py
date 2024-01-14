#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:42:15 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

traindf=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/train.csv',index_col=0)

xtrain=traindf.drop('TARGET',axis=1)
ytrain=traindf['TARGET']

xtest= pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/test.csv',index_col=0)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

ytrain.value_counts().plot(kind='bar')

####### gaussian NB ##############

gnb=GaussianNB()

gnb.fit(xtrain,ytrain)

result=cross_val_score(gnb,xtrain,ytrain,scoring='roc_auc',cv=kfold)

print(result)
print(result.mean())

y_pred_prob1=gnb.predict_proba(xtest)[:,1]
y_pred=gnb.predict(xtest)

####### Logistic Regression ##############

logreg=LogisticRegression()

logreg.fit(xtrain,ytrain)

result=cross_val_score(logreg,xtrain,ytrain,scoring='roc_auc',cv=kfold)

print(result)
print(result.mean())

y_pred_prob=logreg.predict_proba(xtest)[:,1]
y_pred=logreg.predict(xtest)

y_pred.value_counts()

####### KNN ##############

knn=KNeighborsClassifier()

pipe=Pipeline([('knn_model',knn)])

print(pipe.get_params())

params={'knn_model__n_neighbors': np.arange(1,30,2)}

gcv=GridSearchCV(pipe, param_grid=params,scoring='roc_auc',cv=kfold)

gcv.fit(xtrain,ytrain)

print(gcv.best_params_)
print(gcv.best_score_)

knn=KNeighborsClassifier(n_neighbors=29)

knn.fit(xtrain,ytrain)

y_pred_prob2=logreg.predict_proba(xtest)[:,1]
y_pred=logreg.predict(xtest)

y_pred.value_counts()

####### Submition for KNN ##############

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob2

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sub_result_knn.csv')

####### Submition for Logistic Regression ##############

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sub_result.csv')

####### Submition for Gaussian NB ##############

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob1

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sub_result_gnb.csv')

