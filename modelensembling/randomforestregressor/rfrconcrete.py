#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:10:06 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

########### concrete strength RandomForestRegressor ################

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']
clf =  RandomForestRegressor(random_state=2022)

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
params = {"max_features":[2,3,4,5,6]}

gcv = GridSearchCV(clf, param_grid=params,cv=kfold,scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)


concrete_test=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/testConcrete.csv')
y_pred=gcv.predict(concrete_test)
concrete_test['Strength']=y_pred
concrete_test.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/modelensembling/randomforestclassifier/rfrconcrete_test.csv')

gcv_model=gcv.best_estimator_

plt.figure(figsize=(40,30))
print(gcv_model.feature_importances_)
ind = np.arange(x.shape[1])
plt.bar(ind,gcv_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()

imp = gcv_model.feature_importances_

i_sorted = np.argsort(-imp)
col_sorted = x.columns[i_sorted]
imp_sorted = imp[i_sorted]

ind = np.arange(x.shape[1])

plt.figure(figsize=(40,30))
plt.bar(ind,imp_sorted)
plt.xticks(ind,x.columns,rotation=90,fontsize=25)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()

########### concrete strength DecisionTreeRegressor ################

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']
clf =  DecisionTreeRegressor(random_state=2022)

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
params = {"max_features":[2,3,4,5,6]}

gcv = GridSearchCV(clf, param_grid=params,cv=kfold,scoring='r2')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

gcv_model=gcv.best_estimator_

plt.figure(figsize=(40,30))
print(gcv_model.feature_importances_)
ind = np.arange(x.shape[1])
plt.bar(ind,gcv_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90,fontsize=25)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()