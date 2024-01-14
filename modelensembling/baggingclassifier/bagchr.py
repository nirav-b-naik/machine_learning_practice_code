#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 08:43:27 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics')

hr= pd.read_csv('HR_comma_sep.csv')

d_hr=pd.get_dummies(hr,drop_first=True)

x=d_hr.drop('left',axis=1)

y=d_hr['left']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           stratify=y,
                                           random_state=2022)

################################## GaussianNB and Bagging

nb = GaussianNB()

bagging = BaggingClassifier(base_estimator=nb,random_state=2022)

bagging.fit(xtrain, ytrain)

y_pred_prob = bagging.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) #0.8126002223578261

################################## GaussianNB only

nb.fit(xtrain, ytrain)

y_pred_prob = nb.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) # 0.8161787510765948

################################## LogisticRegression and Bagging

logreg = LogisticRegression()

bagging = BaggingClassifier(base_estimator=logreg,random_state=2022)

bagging.fit(xtrain, ytrain)

y_pred_prob = bagging.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) #0.8144466691118948

################################## LogisticRegression only

logreg.fit(xtrain, ytrain)

y_pred_prob = logreg.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) #0.8124286751737732

################################## DecisionTreeClassifier and Bagging

clf = DecisionTreeClassifier(random_state=2022)

bagging = BaggingClassifier(base_estimator=clf,random_state=2022)

bagging.fit(xtrain, ytrain)

y_pred_prob = bagging.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) # 0.9852732188432873

################################## DecisionTreeClassifier only

clf.fit(xtrain, ytrain)

y_pred_prob = clf.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) # 0.972023104955018
