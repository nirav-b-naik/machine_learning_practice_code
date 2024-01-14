#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:17:15 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/bank/bank.csv", sep=";")
df_dummy = pd.get_dummies(df,drop_first=True)

x = df_dummy.drop("y_yes",axis=1)
y = df_dummy["y_yes"]

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

rf = RandomForestClassifier(random_state=2022)

params = {"max_depth":[None,2,3],
          "max_features":[2,3,4],
          'min_samples_leaf': [5,10],
          'min_samples_split': [5,10]}

gcv = GridSearchCV(rf, param_grid=params,scoring="roc_auc",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
# {'max_depth': None, 'max_features': 4, 'min_samples_leaf': 5, 'min_samples_split': 5}
print(gcv.best_score_)
# 0.8979813644688646

best_model = gcv.best_estimator_
imp = best_model.feature_importances_
i_soft = np.argsort(-imp)
sorted_imp = imp[i_soft]
sorted_col = x.columns[i_soft]

ind = np.arange(x.shape[1])
plt.bar(ind,sorted_imp)
plt.xticks(ind,(sorted_col),rotation=90)
plt.title('feature importance')
plt.xlabel('variables')


########################################### GradientBoosting
gbm = GradientBoostingClassifier(random_state=2022)
params_gbm = {"learning_rate":np.linspace(0.001,0.5,10),
          "max_depth":[2,3,4,5,6],
          "n_estimators":[50,100,150]}

gcv = GridSearchCV(gbm, param_grid=params,scoring="roc_auc",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
# {'max_depth': 3, 'max_features': 4, 'min_samples_leaf': 5, 'min_samples_split': 5}
print(gcv.best_score_)
# 0.9051359661172163
