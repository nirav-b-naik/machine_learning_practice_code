#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:54:42 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression,SGDClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.naive_bayes import GaussianNB

import os

os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/ottogroup")

train = pd.read_csv("train.csv",index_col=0)
test = pd.read_csv("test.csv",index_col=0)

x = train.drop("target",axis=1)
y =train["target"]

le = LabelEncoder()
le_y  = le.fit_transform(y)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
scaler = MinMaxScaler()

############################### Gradient Descent ############################################

sgd = SGDClassifier(loss='log_loss')
#sgd.fit(x, le_y)
pipe= Pipeline([('scl_mm',scaler),('sgd_model',sgd)])

result = cross_val_score(pipe,x,le_y,cv=kfold,scoring="neg_log_loss")

print(result)
print(result.mean())

############################### GaussianNB ############################################
sgd = SGDClassifier(loss='log_loss')
#sgd.fit(x, le_y)
pipe= Pipeline([('scl_mm',scaler),('sgd_model',sgd)])
params={'sgd_model__eta0':np.linspace(0.001, 0.7,5),'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive']}

gcv=GridSearchCV(pipe, param_grid=params,scoring='neg_log_loss',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model=gcv.best_estimator_

y_pred_prob=best_model.predict_proba(test)

pd_pred_prob=pd.DataFrame(y_pred_prob,columns=list(le.classes_))

## Submission
submit = pd.read_csv("sampleSubmission.csv")
submission = pd.concat([submit['id'],pd_pred_prob],axis=1)

submission.to_csv("submit_sgd.csv",index=False)
