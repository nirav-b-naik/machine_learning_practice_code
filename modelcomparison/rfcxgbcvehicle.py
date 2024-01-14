#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:00:11 2022

@author: dai
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Vehicle Silhouettes/Vehicle.csv")
x = df.drop("Class",axis=1)
y = df["Class"]

le = LabelEncoder()
le_y  =le.fit_transform(y)


#####################XGBClassifier

xgb = XGBClassifier(random_state=2022, objective='multi:softprob')
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

params = {"learning_rate":np.linspace(0.01,0.5,10),
          "max_depth":[2,3,4],
          "n_estimators":[20,50]}

gcv = GridSearchCV(xgb, param_grid=params,scoring="neg_log_loss",cv=kfold,verbose=10)
gcv.fit(x,le_y)

print(gcv.best_params_)
# {'learning_rate': 0.44555555555555554, 'max_depth': 2, 'n_estimators': 50}

print(gcv.best_score_)
# -0.4638789065170288

print(le.classes_)

####################################### Random Forest
rf =RandomForestClassifier(random_state=2022)

params = {"max_features":[2,3,4,5,6]}

gcv = GridSearchCV(rf, param_grid=params,scoring="neg_log_loss",cv=kfold,verbose=10)

gcv.fit(x,le_y)

print(gcv.best_params_)
# {'max_depth': 4, 'max_features': 6, 'n_estimators': 50}

print(gcv.best_score_)
# -0.6509829997370242

#####################XGBRFClassifier

xgb = XGBRFClassifier(random_state=2022, objective='multi:softprob')

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

params = {"learning_rate":np.linspace(0.01,0.5,10),
          "max_depth":[2,3,4],
          "n_estimators":[20,50]}

gcv = GridSearchCV(xgb, param_grid=params,scoring="neg_log_loss",cv=kfold,verbose=10)
gcv.fit(x,le_y)

print(gcv.best_params_)
# 
'''{'learning_rate': 0.5, 'max_depth': 4, 'n_estimators': 20}
-0.9742122359011095'''
print(gcv.best_score_)
# 

print(le.classes_)



