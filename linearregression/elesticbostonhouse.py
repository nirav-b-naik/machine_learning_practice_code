#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:14:57 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.linear_model import ElasticNet
import os
import kaggle

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning')

!kaggle datasets download -d hellbuoy/car-price-prediction

!kaggle datasets download -d kukuroo3/used-car-price-dataset-competition-format

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing/Boston.csv")
x = df.drop('medv',axis=1)
y = df['medv']

kfold = KFold(n_splits=5,shuffle=True,random_state=True)
lr = ElasticNet()
params = {'alpha':np.linspace(0.001,1000),'l1_ratio':np.linspace(0,1)}
gcv = GridSearchCV(lr, param_grid=params,cv=kfold,scoring='r2')
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
