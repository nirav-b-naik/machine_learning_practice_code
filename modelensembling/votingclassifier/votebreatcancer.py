#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:00:18 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin')

cancer= pd.read_csv('BreastCancer.csv')

d_cancer=pd.get_dummies(cancer,drop_first=True)

x=d_cancer.drop('Class_Malignant',axis=1)

y=d_cancer['Class_Malignant']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           stratify=y,
                                           random_state=2022)

clf=DecisionTreeClassifier(random_state=2022)
nb = GaussianNB()
svm =SVC(probability=True,random_state=2022)
bnb=BernoulliNB()
logreg=LogisticRegression()

models = [("Tree",clf),("Naive",nb),("SVM",svm),("BNB",bnb),("LOGREG",logreg)]

voting = VotingClassifier(models,voting='soft')
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(accuracy_score(ytest, y_pred))

y_pred_prob = voting.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob))
