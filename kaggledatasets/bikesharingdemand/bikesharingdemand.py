#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:20:38 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/bikesharingdemand')

# convertig datetime columns in train set
train=pd.read_csv('train.csv',parse_dates=['datetime'])

train['year']=train['datetime'].dt.year
train['month']=train['datetime'].dt.month
train['day']=train['datetime'].dt.day
train['hour']=train['datetime'].dt.hour
train['weekday']=train['datetime'].dt.weekday

train=train.set_index('datetime')

# convertig datetime columns in test set
test=pd.read_csv('test.csv',parse_dates=['datetime'])

test['year']=test['datetime'].dt.year
test['month']=test['datetime'].dt.month
test['day']=test['datetime'].dt.day
test['hour']=test['datetime'].dt.hour
test['weekday']=test['datetime'].dt.weekday

test=test.set_index('datetime')

x=train.drop(['casual', 'registered', 'count'],axis=1)
y=train['count']

kfold=KFold(n_splits=5,shuffle=True, random_state=2022)

lr=LinearRegression()
lr.fit(x,y)

result=cross_val_score(lr,x,y,scoring='r2',cv=kfold,verbose=3)

print(result.mean())
print(lr.coef_)
print(lr.intercept_)

ypred=lr.predict(test)

ypred[ypred<0]=0

######## submit

submit=pd.read_csv('sampleSubmission.csv',index_col=0)

submit['count']=ypred

submit.to_csv('bike_lr_2.csv')

############### relation and plotting

import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

ols_count=ols('count ~ weekday',data=train).fit()
table=anova_lm(ols_count, typ=2)
print(table)

ols_reg=ols('registered ~ weekday',data=train).fit()
table=anova_lm(ols_reg, typ=2)
print(table)

ols_casual=ols('casual ~ weekday',data=train).fit()
table=anova_lm(ols_casual, typ=2)
print(table)

plt.figure()
train.groupby(['weekday'])['count'].mean().plot(kind='bar')
plt.title('count ~ weekday')

plt.figure()
train.groupby(['weekday'])['registered'].mean().plot(kind='bar')
plt.title('registered ~ weekday')

plt.figure()

train.groupby(['weekday'])['casual'].mean().plot(kind='bar')
plt.title('casual ~ weekday')

########## y predict for casual

x=train.drop(['casual', 'registered', 'count'],axis=1)
y_casual=train['casual']

kfold=KFold(n_splits=5,shuffle=True, random_state=2022)

lr_casual=LinearRegression()
lr_casual.fit(x,y_casual)

result=cross_val_score(lr_casual,x,y_casual,scoring='r2',cv=kfold,verbose=3)

print(result.mean())
print(lr_casual.coef_)
print(lr_casual.intercept_)

ypred_casual=lr_casual.predict(test)

ypred_casual[ypred_casual<0]=0


########## y predict for registered

x=train.drop(['casual', 'registered', 'count'],axis=1)
y_reg=train['registered']

kfold=KFold(n_splits=5,shuffle=True, random_state=2022)

lr_reg=LinearRegression()
lr_reg.fit(x,y_reg)

result=cross_val_score(lr_reg,x,y_reg,scoring='r2',cv=kfold,verbose=3)

print(result.mean())
print(lr_reg.coef_)
print(lr_reg.intercept_)

ypred_registered=lr_reg.predict(test)

ypred_registered[ypred_registered<0]=0


############# 

ypred_count=ypred_casual+ypred_registered

######## submit

submit=pd.read_csv('sampleSubmission.csv',index_col=0)

submit['count']=ypred_count

submit.to_csv('bike_lr_4.csv')
