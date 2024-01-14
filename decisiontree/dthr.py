#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:44:23 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics/HR_comma_sep.csv")
df_dummy = pd.get_dummies(df,drop_first=True)
x = df_dummy.drop('left',axis=1)
y = df_dummy["left"]

clf =  DecisionTreeClassifier(random_state=2022)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {"max_depth":[None,10,5,3],
          "min_samples_split":[2,10,50,100],
          "min_samples_leaf":[1,10,50,100]}

gcv = GridSearchCV(clf, param_grid=params,cv=kfold,scoring='roc_auc')
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

gcv_model=gcv.best_estimator_

plt.figure(figsize=(400,300))
plot_tree(gcv_model,feature_names=x.columns,
          class_names=['stayed','left'],
          filled=True,fontsize=10)

plt.figure()
print(gcv_model.feature_importances_)
ind = np.arange(18)
plt.bar(ind,gcv_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()
