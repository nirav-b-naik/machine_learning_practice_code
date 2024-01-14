#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:52:07 2022

@author: dai
"""

import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Satellite Imaging/Satellite.csv",sep=';')
x = df.drop("classes",axis=1)
y = df["classes"]

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
