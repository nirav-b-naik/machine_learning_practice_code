#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:06:12 2022

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

traindf=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/train.csv',index_col=0)

xtrain=traindf.drop('TARGET',axis=1)
ytrain=traindf['TARGET']

xtest= pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/test.csv',index_col=0)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

####### LinearDiscriminantAnalysis ##############

da=LinearDiscriminantAnalysis()

da.fit(xtrain,ytrain)

result=cross_val_score(da,xtrain,ytrain,scoring='roc_auc',cv=kfold)

print(result)
print(result.mean())

y_pred_prob1=da.predict_proba(xtest)[:,1]
y_pred=da.predict(xtest)

##############submission

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob1

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sub_result_da.csv')

####### QuadraticDiscriminantAnalysis ##############

qda=QuadraticDiscriminantAnalysis()

qda.fit(xtrain,ytrain)

result=cross_val_score(qda,xtrain,ytrain,scoring='roc_auc',cv=kfold)

print(result)
print(result.mean())

y_pred_prob2=qda.predict_proba(xtest)[:,1]
y_pred=qda.predict(xtest)

##############submission

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob2

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sub_result_qda.csv')



