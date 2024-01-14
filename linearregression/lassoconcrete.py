#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:25:42 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)

lr=Lasso()

lr.fit(x_train,y_train)

print(lr.intercept_)
print(lr.coef_)

y_pred=lr.predict(x_test)
print(r2_score(y_test, y_pred))

##########################GridsearchCV##########################

params={'alpha': np.linspace(0.01,1000)}

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

gcv=GridSearchCV(lr, param_grid=params,scoring='r2',cv=kfold,verbose=3)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
