#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:29:12 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Vehicle Silhouettes/Vehicle.csv")
x = df.drop("Class",axis=1)
y = df["Class"]

kfold =  StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

le = LabelEncoder()
le_y = le.fit_transform(y)

###########################LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(x,le_y)

result = cross_val_score(lda, x,le_y,cv=kfold,scoring='neg_log_loss')

print(result)
print(result.mean())

###########################QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(x,le_y)

result = cross_val_score(qda, x,le_y,cv=kfold,scoring='neg_log_loss')

print(result)
print(result.mean())
