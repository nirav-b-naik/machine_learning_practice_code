#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:01:56 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV,train_test_split,KFold
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor

df= pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Medical Cost Personal/insurance.csv")
df_dum = pd.get_dummies(df,drop_first=True)

x =df_dum.drop("charges",axis=1)
y = df_dum["charges"]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           random_state=2022)

lr = LinearRegression()
en = ElasticNet()
clf = DecisionTreeRegressor()
rlr = Ridge()
llr = Lasso()

#########################################################
models = [("en",en),("lr",lr),("clf",clf)]
voting = VotingRegressor(models,verbose=1)
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(r2_score(ytest, y_pred)) #0.7772336057745673

#########################################################
models = [("rlr",rlr),("lr",lr),("clf",clf)]
voting = VotingRegressor(models,verbose=1)
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(r2_score(ytest, y_pred)) #0.82166873683654

#########################################################
models = [("llr",llr),("lr",lr),("clf",clf)]
voting = VotingRegressor(models,verbose=1)
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(r2_score(ytest, y_pred)) #0.8219558417169381

#######################################################

lr.fit(xtrain,ytrain)

y_pred = lr.predict(xtest)

r2_lr=r2_score(ytest, y_pred)

print(r2_score(ytest, y_pred)) #0.7815638027456551

#######################################################

en.fit(xtrain,ytrain)

y_pred = en.predict(xtest)

r2_en=r2_score(ytest, y_pred)

print(r2_score(ytest, y_pred)) #0.41420291743879134

#######################################################

clf.fit(xtrain,ytrain)

y_pred = clf.predict(xtest)

r2_clf=r2_score(ytest, y_pred)

print(r2_score(ytest, y_pred)) #0.754129999363216

###################with weight

models = [("en",en),("lr",lr),("clf",clf)]
voting = VotingRegressor(models,weights=np.array([r2_en,r2_lr,r2_clf]),verbose=1)
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(r2_score(ytest, y_pred)) #0.8029787714896849

##################with grid search and params

kfold=KFold(n_splits=5,shuffle=True, random_state=2022)
models = [("en",en),("lr",lr),("clf",clf)]
voting = VotingRegressor(models)

params={'en__alpha':[0.5,1,1.5],
        'en__l1_ratio': [0.25,0.5,0.75],
        'clf__max_depth': [None,2,3,4,5,6,7,8,9,10],
        'clf__min_samples_leaf': [2,5,10],
        'clf__min_samples_split': [2,5,10]}

gcv=GridSearchCV(voting,param_grid=params,scoring='r2',cv=kfold,verbose=1)

gcv.fit(x,y)

print(gcv.best_params_)
'''
{'clf__max_depth': None, 
 'clf__min_samples_leaf': 2, 
 'clf__min_samples_split': 5, 
 'en__alpha': 1, 
 'en__l1_ratio': 0.5}
'''
print(gcv.best_score_) #0.7791682871364578




