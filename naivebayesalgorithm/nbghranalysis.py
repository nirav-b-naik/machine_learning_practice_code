#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:29:53 2022

@author: dai
"""


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics/HR_comma_sep.csv")

dummy_df = pd.get_dummies(df,drop_first=True)
x= dummy_df.drop('left',axis=1)
y = dummy_df['left']

gnb = GaussianNB()
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
result = cross_val_score(gnb, x,y,cv=kfold,scoring='roc_auc')
print(result)

print(result.mean())