#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:44:07 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score


os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Telecom')

df=pd.read_csv('Telecom.csv')

dum_tel= pd.get_dummies(df,drop_first=True)

x=dum_tel.drop('Response_Y',axis=1)
y=dum_tel['Response_Y']

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.3,
                                               random_state=2022,
                                               stratify=y)

nb=BernoulliNB()

nb.fit(x_train,y_train)

y_pred_prob=nb.predict_proba(x_test)[:,1]

print(roc_auc_score(y_test,y_pred_prob))

## on test file

test_tele=pd.read_csv('testTelecom.csv')
dum_test= pd.get_dummies(test_tele,drop_first=True)

predict_prob=nb.predict_proba(dum_test)
prediction=nb.predict(dum_test)
