#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:20:12 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Sonar')
df=pd.read_csv('Sonar.csv')
dummy_sonar=pd.get_dummies(df,drop_first=True)
x=dummy_sonar.drop('Class_R',axis=1)
y=dummy_sonar['Class_R']

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2022)

gnb=GaussianNB()

result=cross_val_score(gnb,x,y,scoring='roc_auc',cv=kfold)

print(result)
print(result.mean())
