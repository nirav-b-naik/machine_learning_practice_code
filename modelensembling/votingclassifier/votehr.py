#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:25:55 2022

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics')

hr= pd.read_csv('HR_comma_sep.csv')

d_hr=pd.get_dummies(hr,drop_first=True)

x=d_hr.drop('left',axis=1)

y=d_hr['left']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           stratify=y,
                                           random_state=2022)

clf = DecisionTreeClassifier(random_state=2022)
nb = GaussianNB()
svm_l = SVC(probability=True,kernel='linear',random_state=2022)
svm_r = SVC(probability=True,kernel='rbf',random_state=2022)
bnb = BernoulliNB()
logreg = LogisticRegression(max_iter=1000)
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

################ Model1: tree, svm_l, lda

model1 = [("Tree",clf),("SVML",svm_l),("LDA",lda)]

voting1 = VotingClassifier(model1,voting='soft',verbose=1)
voting1.fit(xtrain,ytrain)

y_pred1 = voting1.predict(xtest)
print(accuracy_score(ytest, y_pred1)) #0.9122222222222223

y_pred_prob1 = voting1.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob1)) #0.9726771626313595

################ Model2: tree, svm_r, qda

model2 = [("Tree",clf),("SVMR",svm_r),("QDA",qda)]

voting2 = VotingClassifier(model2,voting='soft')
voting2.fit(xtrain,ytrain)

y_pred2 = voting2.predict(xtest)
print(accuracy_score(ytest, y_pred2)) #0.9591111111111111

y_pred_prob2 = voting2.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob2)) #0.9838113917677501

################ Model3: logreg, nb, lda

model3 = [("LOGREG",logreg),("NB",nb),("LDA",lda)]

voting3 = VotingClassifier(model3,voting='soft')
voting3.fit(xtrain,ytrain)

y_pred3 = voting3.predict(xtest)
print(accuracy_score(ytest, y_pred3)) #0.8182222222222222

y_pred_prob3 = voting3.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob3)) #0.8299384690203485
