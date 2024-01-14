#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:55:44 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics/HR_comma_sep.csv")
df_dummy = pd.get_dummies(df,drop_first=True)
x = df_dummy.drop('left',axis=1)
y = df_dummy["left"]

clf =  RandomForestClassifier(random_state=2022)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {"max_features":[2,3,4,5,6]}

gcv = GridSearchCV(clf, param_grid=params,cv=kfold,scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_) # 'max_features': 6
print(gcv.best_score_) # 0.9940818009455782

gcv_model=gcv.best_estimator_

plt.figure(figsize=(400,300))
plot_tree(gcv_model,feature_names=x.columns,
          class_names=['stayed','left'],
          filled=True,fontsize=10)

plt.figure(figsize=(40,30))
print(gcv_model.feature_importances_)
ind = np.arange(18)
plt.bar(ind,gcv_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()