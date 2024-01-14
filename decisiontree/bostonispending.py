#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:09:23 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.metrics import r2_score

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing/Boston.csv")
x = df.drop("medv",axis=1)
y = df["medv"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=2022)

clf = DecisionTreeRegressor(random_state=2022)
clf.fit(xtrain,ytrain)

y_pred = clf.predict(xtest)
