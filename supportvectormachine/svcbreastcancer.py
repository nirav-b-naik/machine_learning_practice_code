#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:25:20 2022

@author: dai
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

cancer = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin/BreastCancer.csv",index_col=0)
dum_cancer = pd.get_dummies(cancer,drop_first=True)
x = dum_cancer.drop('Class_Malignant',axis=1)
y = dum_cancer["Class_Malignant"]
xtrain,xtest,ytrain,ytest = train_test_split(x,
                                             y,
                                             test_size=0.3,
                                             stratify=y,
                                             random_state=2022)
svm = SVC(probability=True)
svm.fit(xtrain,ytrain)

y_pred_prob = svm.predict_proba(xtest)[:,1]
print(roc_auc_score(ytest, y_pred_prob))

###################### GridSearchCV with RBF############################################

svm = SVC(probability=True,kernel='rbf')
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'C':np.linspace(0.001,5,10),'gamma':np.linspace(0.001,5,10)}
gsv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc',verbose=1)
gsv.fit(x,y)
print(gsv.best_params_)
print(gsv.best_score_)

###################### GridSearchCV with polynomial############################################

svm = SVC(probability=True,kernel='poly')
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'C':np.linspace(0.001,5,10),'gamma':np.linspace(0.001,5,10), 'degree': [2,3,4]}
gsv = GridSearchCV(svm, param_grid=params,cv=kfold,scoring='roc_auc',verbose=1)
gsv.fit(x,y)
print(gsv.best_params_)
print(gsv.best_score_)

