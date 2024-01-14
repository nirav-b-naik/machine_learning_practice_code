#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:34:40 2022

@author: dai
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold,KFold
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Chemical Process Data/ChemicalProcess.csv")
x = df.drop("Yield",axis=1)
y = df["Yield"]

##################################### Using median as imputer
imp = SimpleImputer(strategy='median')

xgb = XGBRegressor(random_state=2022)
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

pipe = Pipeline([("imp",imp),("xgb",xgb)])
params = {"xgb__learning_rate":np.linspace(0.001,1,10),
          "xgb__max_depth":[2,3,4,5,6],
          "xgb__n_estimators":[50,100,150]}

gcv = GridSearchCV(pipe, param_grid=params,scoring="r2",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
# {'xgb__learning_rate': 0.223, 'xgb__max_depth': 2, 'xgb__n_estimators': 100}
print(gcv.best_score_) # 0.6812725073299442

df_cv = pd.DataFrame(gcv.cv_results_)

##################################### Using mean as imputer
imp = SimpleImputer(strategy='mean')
pipe = Pipeline([("imp",imp),("xgb",xgb)])

gcv = GridSearchCV(pipe, param_grid=params,scoring="r2",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
# {'xgb__learning_rate': 0.445, 'xgb__max_depth': 3, 'xgb__n_estimators': 50}
print(gcv.best_score_) # 0.6948941432676101

df_cv = pd.DataFrame(gcv.cv_results_)