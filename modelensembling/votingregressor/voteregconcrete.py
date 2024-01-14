#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:32:20 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")
x =df.drop("Strength",axis=1)
y = df["Strength"]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           random_state=2022)

en = ElasticNet()
lr = LinearRegression()
clf = DecisionTreeRegressor()

models = [("en",en),("lr",lr),("clf",clf)]
voting = VotingRegressor(models,verbose=1)
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(r2_score(ytest, y_pred)) # 0.7924359873417388 

############################################################

en.fit(xtrain,ytrain)
y_pred = en.predict(xtest)
r2_en = r2_score(ytest,y_pred)
print(r2_score(ytest,y_pred)) #0.6465499867835319

############################################################

lr.fit(xtrain,ytrain)
y_pred = lr.predict(xtest)
r2_lr = r2_score(ytest,y_pred)
print(r2_score(ytest,y_pred)) #0.647164752566981
############################################################

clf.fit(xtrain,ytrain)
y_pred = clf.predict(xtest)
r2_clf = r2_score(ytest,y_pred)
print(r2_score(ytest,y_pred)) #0.8003670322032621

################################ WIth Weight 

models = [("en",en),("lr",lr),("clf",clf)]
voting = VotingRegressor(models,weights=np.array([r2_en,r2_lr,r2_clf]),verbose=1)
voting.fit(xtrain,ytrain)

y_pred = voting.predict(xtest)
print(r2_score(ytest, y_pred)) # 0.8074369425246751

