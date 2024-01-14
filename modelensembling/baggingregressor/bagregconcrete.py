#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:06:51 2022

@author: dai
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor
########### concrete strength 

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

clf=DecisionTreeRegressor(random_state=2022)

params= {'max_depth':[None,3,5],
         'min_samples_leaf':[1,5,10],
         'min_samples_split':[2,5,10]}

gcv=GridSearchCV(clf, param_grid=params,cv=kfold,scoring='r2',verbose=1)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
# {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5}
# 0.8499614669673818

########### concrete strength with bagging

bag=BaggingRegressor(base_estimator=clf,random_state=2022)

params= {'base_estimator__max_depth':[None,3,5],
         'base_estimator__min_samples_leaf':[1,5,10],
         'base_estimator__min_samples_split':[2,5,10],
         'n_estimators':[10,50,100]}

gcv=GridSearchCV(bag, param_grid=params,cv=kfold,scoring='r2',verbose=1)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)
# {'base_estimator__max_depth': None, 'base_estimator__min_samples_leaf': 1,
# 'base_estimator__min_samples_split': 2, 'n_estimators': 100}
# 0.917073492771132
