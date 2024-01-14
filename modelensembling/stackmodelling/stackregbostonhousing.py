#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:40:14 2022
console:2
@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold,GridSearchCV, train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing')

boston=pd.read_csv('Boston.csv')

x=boston.drop('medv',axis=1)
y=boston['medv']

xgb=XGBRegressor()

lr=LinearRegression()

knn= KNeighborsRegressor()

dtr=DecisionTreeRegressor(random_state=2022)

rfr=RandomForestRegressor(random_state=2022)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2022)

models=[('lr',lr),('dtr',dtr),('knn',knn)]

#################### with final_estimator=XGBRegressor

stack=StackingRegressor(estimators=models,final_estimator=xgb,passthrough=True)

stack.fit(xtrain,ytrain)

y_pred=stack.predict(xtest)

print(r2_score(ytest, y_pred))
# 0.7613461700656149 - without passthrough
# 0.8088627370381494 - with passthrough

#################### with final_estimator=RandomForestRegressor
stack=StackingRegressor(estimators=models,final_estimator=rfr,passthrough=True)

stack.fit(xtrain,ytrain)

y_pred=stack.predict(xtest)

print(r2_score(ytest, y_pred))
# 0.8085462009708173 - without passthrough
# 0.8363279581069234 - with passthrough

#################### with final_estimator=RandomForestRegressor with gridsearchcv

kfold=KFold(n_splits=5,shuffle=True, random_state=2022)

stack=StackingRegressor(estimators=models,final_estimator=rfr,passthrough=True)

params={'final_estimator__max_features': 1.0,
        'dtr__max_depth': None,
        }

gcv=GridSearchCV(stack, param_grid=params,scoring='r2',cv=kfold,verbose=1)

gcv.fit(x,y)

print(gcv.best_params_)
#

print(gcv.best_score_)
#

