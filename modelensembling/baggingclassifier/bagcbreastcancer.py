#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 07:39:50 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin')

cancer= pd.read_csv('BreastCancer.csv',index_col=0)

d_cancer=pd.get_dummies(cancer,drop_first=True)

x=d_cancer.drop('Class_Malignant',axis=1)

y=d_cancer['Class_Malignant']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           stratify=y,
                                           random_state=2022)

################################## GaussianNB and Bagging

nb = GaussianNB()

bagging = BaggingClassifier(base_estimator=nb,random_state=2022)

bagging.fit(xtrain, ytrain)

y_pred_prob = bagging.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) #0.9831924315619968

################################## GaussianNB only

nb.fit(xtrain, ytrain)

y_pred_prob = nb.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) # 0.9810789049919485

################################## DecisionTreeClassifier and Bagging

clf = DecisionTreeClassifier(random_state=2022)

bagging = BaggingClassifier(base_estimator=clf,random_state=2022)

bagging.fit(xtrain, ytrain)

y_pred_prob = bagging.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) # 0.9796698872785828


################################## DecisionTreeClassifier only

clf.fit(xtrain, ytrain)

y_pred_prob = clf.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred_prob)) # 0.9263285024154588

########################### with for loop

for i in range(1,10):
    
    i=i*10000
    
    clf = DecisionTreeClassifier(random_state=2022)
    
    bagging = BaggingClassifier(base_estimator=clf,random_state=i)
    
    bagging.fit(xtrain, ytrain)
    
    y_pred_prob = bagging.predict_proba(xtest)[:,1]
    
    print(f'{i}: ',roc_auc_score(ytest, y_pred_prob))

'''
10000:  0.9774557165861513
20000:  0.9927033011272142
30000:  0.987218196457327
40000:  0.9844001610305959
50000:  0.9851549919484702
60000:  0.9825382447665058
70000:  0.9873691626409017
80000:  0.9828904991948471
90000:  0.9788144122383254
'''