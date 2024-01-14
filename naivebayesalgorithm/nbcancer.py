#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:30:44 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Cancer/Cancer.csv",index_col=0)

dumm_df = pd.get_dummies(df,drop_first=True)
x = dumm_df.drop('Class_recurrence-events',axis=1)
y = dumm_df['Class_recurrence-events']

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
nb = BernoulliNB()
result = cross_val_score(nb,x,y,scoring='roc_auc',cv=kfold)
print(result)
print(result.mean())