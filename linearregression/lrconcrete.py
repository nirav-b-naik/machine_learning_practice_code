#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:01:39 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

########### concrete strength ################

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)

lr=LinearRegression()

lr.fit(x_train,y_train)

print(lr.intercept_)
print(lr.coef_)

y_pred=lr.predict(x_test)
print(r2_score(y_test, y_pred))

##################Linear regression##########

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

pipe=Pipeline([('lr_model',lr)])

result=cross_val_score(pipe,x,y,scoring='r2',cv=kfold)

print(result)

print(result.mean())

##################forward selection stepwise regression with r2##########

feature=['Cement']
x = b_concrete[feature]
y = b_concrete['Strength']
   
x=x.values
x=x.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)

lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
print(f'{feature}:r2: ',r2_score(y_test, y_pred))

allfeature= ['Blast', 'Fly', 'Water', 'Superplasticizer', 'Coarse', 'Fine','Age']

for col in allfeature:
    
    feature.append(col)

    x = b_concrete[feature]
    y = b_concrete['Strength']
       
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)
    
    lr=LinearRegression()
    
    lr.fit(x_train,y_train)
    
    y_pred=lr.predict(x_test)
    print(f'{feature}:r2: ',r2_score(y_test, y_pred))
    
##################forward selection stepwise regression with mean_absolute_error ##########

feature=['Cement']
x = b_concrete[feature]
y = b_concrete['Strength']
   
x=x.values
x=x.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)

lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
print(f'{feature}:mean_absolute_error: ',mean_absolute_error(y_test, y_pred))

allfeature= ['Blast', 'Fly', 'Water', 'Superplasticizer', 'Coarse', 'Fine','Age']

for col in allfeature:
    
    feature.append(col)

    x = b_concrete[feature]
    y = b_concrete['Strength']
       
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)
    
    lr=LinearRegression()
    
    lr.fit(x_train,y_train)
    
    y_pred=lr.predict(x_test)
    print(f'{feature}:mean_absolute_error: ',mean_absolute_error(y_test, y_pred))
    
##################forward selection stepwise regression with mean_absolute_error ##########

feature=['Cement']
x = b_concrete[feature]
y = b_concrete['Strength']
   
x=x.values
x=x.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)

lr=LinearRegression()

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
print(f'{feature}:mean_squared_error: ',mean_squared_error(y_test, y_pred))

allfeature= ['Blast', 'Fly', 'Water', 'Superplasticizer', 'Coarse', 'Fine','Age']

for col in allfeature:
    
    feature.append(col)

    x = b_concrete[feature]
    y = b_concrete['Strength']
       
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)
    
    lr=LinearRegression()
    
    lr.fit(x_train,y_train)
    
    y_pred=lr.predict(x_test)
    print(f'{feature}:mean_squared_error: ',mean_squared_error(y_test, y_pred))
    
##################single selection stepwise regression with r2##########


for col in b_concrete.columns:
    
    x = b_concrete[col]
    y = b_concrete['Strength']
       
    x=x.values
    x=x.reshape(-1,1)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2022)
    
    lr=LinearRegression()
    
    lr.fit(x_train,y_train)
    
    y_pred=lr.predict(x_test)
    print(f'{col}:r2: ',r2_score(y_test, y_pred))
    
    
    
    





