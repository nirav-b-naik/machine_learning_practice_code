#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:54:42 2022

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

os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/ottogroup")

train = pd.read_csv("train.csv",index_col=0)
test = pd.read_csv("test.csv",index_col=0)

x = train.drop("target",axis=1)
y =train["target"]

le = LabelEncoder()
le_y  = le.fit_transform(y)

############################### LogisticRegression ############################################
lr = LogisticRegression(multi_class="ovr")
lr.fit(x, le_y)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

result = cross_val_score(lr,x,le_y,cv=kfold,scoring="neg_log_loss")

print(result)
print(result.mean())

y_pred_prob = lr.predict_proba(test)

pd_pred_prob = pd.DataFrame(y_pred_prob,columns=list(le.classes_))

## Submission
submit = pd.read_csv("sampleSubmission.csv")
submission = pd.concat([submit['id'],pd_pred_prob],axis=1)

submission.to_csv("submit_log_reg.csv",index=False)

############################### GaussianNB ############################################
gnb = GaussianNB()
gnb.fit(x, le_y)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

result = cross_val_score(gnb,x,le_y,cv=kfold,scoring="neg_log_loss")

print(result)
print(result.mean())

y_pred_prob_gnb = gnb.predict_proba(test)

pd_pred_prob_gnb = pd.DataFrame(y_pred_prob_gnb,columns=list(le.classes_))

## Submission
submit = pd.read_csv("sampleSubmission.csv")
submission = pd.concat([submit['id'],pd_pred_prob_gnb],axis=1)

submission.to_csv("submit_gaussian.csv",index=False)
