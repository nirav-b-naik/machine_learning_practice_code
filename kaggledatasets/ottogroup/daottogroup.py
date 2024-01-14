#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:17:05 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/ottogroup")

train = pd.read_csv("train.csv",index_col=0)
test = pd.read_csv("test.csv",index_col=0)

x = train.drop("target",axis=1)
y =train["target"]

le = LabelEncoder()
le_y  = le.fit_transform(y)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

####### LinearDiscriminantAnalysis ##############

da = LinearDiscriminantAnalysis()
da.fit(x, le_y)

result = cross_val_score(da,x,le_y,cv=kfold,scoring="neg_log_loss")

print(result)
print(result.mean())

y_pred_prob = da.predict_proba(test)

pd_pred_prob = pd.DataFrame(y_pred_prob,columns=list(le.classes_))

## Submission
submit = pd.read_csv("sampleSubmission.csv")
submission = pd.concat([submit['id'],pd_pred_prob],axis=1)

submission.to_csv("submit_da.csv",index=False)

####### QuadraticDiscriminantAnalysis ##############

qda=QuadraticDiscriminantAnalysis()
qda.fit(x, le_y)

result = cross_val_score(qda,x,le_y,cv=kfold,scoring="neg_log_loss")

print(result)
print(result.mean())

y_pred_prob1 = qda.predict_proba(test)

pd_pred_prob1 = pd.DataFrame(y_pred_prob1,columns=list(le.classes_))

## Submission
submit = pd.read_csv("sampleSubmission.csv")
submission = pd.concat([submit['id'],pd_pred_prob1],axis=1)

submission.to_csv("submit_qda.csv",index=False)
