#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:44:41 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing')

df= pd.read_csv('Boston.csv')

x=df.drop('medv',axis=1)
y=df['medv']

scaler=MinMaxScaler()

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

sgd=SGDRegressor(random_state=2022)

poly=PolynomialFeatures(degree=2)
poly_x=poly.fit_transform(x)
print(poly.get_feature_names_out())

pd_poly_x = pd.DataFrame(poly_x,columns=poly.get_feature_names_out())

sgd.fit(pd_poly_x,y)

print(sgd.intercept_)
print(sgd.coef_)

pipe=Pipeline([('scl_mm',scaler),('sgd_model',sgd)])

params={'sgd_model__eta0':np.linspace(0.001,0.7,10),'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive']}

gcv=GridSearchCV(pipe, param_grid=params,scoring='r2',cv=kfold)

gcv.fit(poly_x,y)

print(gcv.best_params_)
print(gcv.best_score_)


############################## poly in pipeline

poly1=PolynomialFeatures()

pipe1=Pipeline([('poly',poly1),('scl_mm',scaler),('sgd_model',sgd)])

params={'sgd_model__eta0':np.linspace(0.001,0.7,10),
        'sgd_model__learning_rate':['constant','optimal','invscaling','adaptive'],
        'poly__degree': [1,2,3,4,5]}

gcv=GridSearchCV(pipe1,param_grid=params,scoring='r2',cv=kfold)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

'''
output:1
{'poly__degree': 4, 'sgd_model__eta0': 0.001, 'sgd_model__learning_rate': 'adaptive'}
0.8259289247412441
'''