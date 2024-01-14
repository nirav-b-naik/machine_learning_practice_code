#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 09:08:10 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Glass Identification')
df = pd.read_csv("Glass.csv")

x=df.drop('Type',axis=1)

y=df['Type']

le = LabelEncoder()
le_y = le.fit_transform(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,le_y,
                                           test_size=0.3,
                                           stratify=y,
                                           random_state=2022)

######################################### GaussianNB and Bagging

nb = GaussianNB()
bagging = BaggingClassifier(base_estimator=nb,random_state=2022)
bagging.fit(xtrain,ytrain)
y_pred = bagging.predict_proba(xtest)
print(log_loss(ytest, y_pred)) # 1.5855373310536378

######################################### Gaussian only

nb = GaussianNB()
nb.fit(xtrain,ytrain)
y_pred = nb.predict_proba(xtest)
print(log_loss(ytest, y_pred)) # 2.9584578737280007

######################################### Logreg and Bagging

logreg = LogisticRegression()
bagging = BaggingClassifier(base_estimator=logreg,random_state=2022)
bagging.fit(xtrain,ytrain)
y_pred = bagging.predict_proba(xtest)
print(log_loss(ytest, y_pred)) # 1.0093327232971214

######################################### LogisticRegression only

logreg = LogisticRegression()
logreg.fit(xtrain,ytrain)
y_pred = logreg.predict_proba(xtest)
print(log_loss(ytest, y_pred)) #1.0288832210950403

######################################### DecisionTreeClassifier and Bagging

clf = DecisionTreeClassifier()
bagging = BaggingClassifier(base_estimator=clf,random_state=2022)
bagging.fit(xtrain,ytrain)
y_pred = bagging.predict_proba(xtest)
print(log_loss(ytest, y_pred)) # 2.1174974391958

######################################### DecisionTreeClassifier only

clf = DecisionTreeClassifier()
clf.fit(xtrain,ytrain)
y_pred = clf.predict_proba(xtest)
print(log_loss(ytest, y_pred)) #10.627315813818678