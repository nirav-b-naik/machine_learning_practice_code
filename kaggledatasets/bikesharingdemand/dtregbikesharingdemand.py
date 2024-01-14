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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/bikesharingdemand')

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

dtr=DecisionTreeRegressor(random_state=2022)

###### DecisionTreeRegressor with cross val score

dtr.fit(x,y)

result=cross_val_score(dtr,x,y,scoring='r2',cv=kfold,verbose=3)

print(result.mean()) #0.8966850944797768

ypred=dtr.predict(test)

ypred[ypred<0]=0

######## submit DecisionTreeRegressor with cross val score

submit=pd.read_csv('sampleSubmission.csv',index_col=0)

submit['count']=ypred

submit.to_csv('bike_dtr_1.csv')

###### DecisionTreeRegressor with GridsearchCV

params={'max_depth': [None,3,8,10],
'min_samples_leaf': np.arange(1,11),
'min_samples_split': np.arange(2,11)}

gcv=GridSearchCV(dtr, param_grid=params,scoring='r2',cv=kfold,verbose=1)

gcv.fit(x,y)

best_model=gcv.best_estimator_

print(gcv.best_params_)
#{'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 10}

print(gcv.best_score_)
#0.91452468193888

ypred=best_model.predict(test)

ypred[ypred<0]=0

######## submit DecisionTreeRegressor with GridsearchCV

submit=pd.read_csv('sampleSubmission.csv',index_col=0)

submit['count']=ypred

submit.to_csv('bike_dtr_2.csv')

########## y predict for casual

x=train.drop(['casual', 'registered', 'count'],axis=1)
y_casual=train['casual']

params={'max_depth': [None,3,8,10],
'min_samples_leaf': np.arange(1,11),
'min_samples_split': np.arange(2,11)}

gcv1=GridSearchCV(dtr,param_grid=params,scoring='r2',cv=kfold,verbose=1)

gcv1.fit(x,y_casual)

best_model1=gcv1.best_estimator_

print(gcv1.best_params_)
#{'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 9}

print(gcv1.best_score_)
#0.8677153251214961

ypred_casual=best_model1.predict(test)

ypred_casual[ypred_casual<0]=0

########## y predict for registered

x=train.drop(['casual', 'registered', 'count'],axis=1)
y_reg=train['registered']

params={'max_depth': [None,3,8,10],
'min_samples_leaf': np.arange(1,11),
'min_samples_split': np.arange(2,11)}

gcv2=GridSearchCV(dtr,param_grid=params,scoring='r2',cv=kfold,verbose=1)

gcv2.fit(x,y_reg)

best_model2=gcv2.best_estimator_

print(gcv2.best_params_)
#{'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 10}

print(gcv2.best_score_)
#0.9284409134250845

ypred_reg=best_model2.predict(test)

ypred_reg[ypred_reg<0]=0

############# 

ypred_count=ypred_casual+ypred_reg

######## submit

submit=pd.read_csv('sampleSubmission.csv',index_col=0)

submit['count']=ypred_count

submit.to_csv('bike_dtr_3.csv')


#############################################################################
#######DecisionTreeRegressor with bagging with gridsearchcv#################
#############################################################################


######### DecisionTreeRegressor with bagging with gridsearchcv y pred_casual

x=train.drop(['casual', 'registered', 'count'],axis=1)
y_casual=train['casual']

bag_c=BaggingRegressor(base_estimator=dtr,random_state=2022)

params={'base_estimator__max_depth': [None,3,8,10],
'base_estimator__min_samples_leaf': np.arange(1,11),
'base_estimator__min_samples_split': np.arange(2,11),
'n_estimators': [10,20,50]}

gcv_c=GridSearchCV(bag_c,param_grid=params,scoring='r2',cv=kfold,verbose=3)

gcv_c.fit(x,y_casual)

best_model_c=gcv_c.best_estimator_

print(gcv_c.best_params_)
# {'base_estimator__max_depth': None, 'base_estimator__min_samples_leaf': 1, 'base_estimator__min_samples_split': 3, 'n_estimators': 50}

print(gcv_c.best_score_)
# 0.9207784910807104

ypred_c=best_model_c.predict(test)



######### DecisionTreeRegressor with bagging with gridsearchcv y pred_reg

x=train.drop(['casual', 'registered', 'count'],axis=1)
y_reg=train['registered']

bag_r=BaggingRegressor(base_estimator=dtr,random_state=2022)

params={'base_estimator__max_depth': [None,3,8,10],
'base_estimator__min_samples_leaf': np.arange(1,11),
'base_estimator__min_samples_split': np.arange(2,11),
'n_estimators': [10,20,50]}

gcv_r=GridSearchCV(bag_r,param_grid=params,scoring='r2',cv=kfold,verbose=3)

gcv_r.fit(x,y_reg)

best_model_r=gcv_r.best_estimator_

print(gcv_r.best_params_)
# {'base_estimator__max_depth': None, 'base_estimator__min_samples_leaf': 1, 'base_estimator__min_samples_split': 3, 'n_estimators': 50}

print(gcv_r.best_score_)
# 0.9521276167556847

ypred_r=best_model_r.predict(test)



############# 

ypred_count2=ypred_c+ypred_r

######## submit

submit=pd.read_csv('sampleSubmission.csv',index_col=0)

submit['count']=ypred_count2

submit.to_csv('bike_dtr_4.csv')


