#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:12:28 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Chemical Process Data/ChemicalProcess.csv")
x = df.drop("Yield",axis=1)
y = df["Yield"]

gbm=XGBRegressor(random_state=2022)

dtr=DecisionTreeRegressor(random_state=2022)

imputer=IterativeImputer(estimator=dtr, max_iter=50, random_state=2022)

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

pipe = Pipeline([("imp",imputer),("xgb",gbm)])
params = {"xgb__learning_rate":np.linspace(0.001,1,10),
          "xgb__max_depth":[2,3,4,5,6],
          "xgb__n_estimators":[50,100,150]}

gcv = GridSearchCV(pipe, param_grid=params,scoring="r2",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
#

print(gcv.best_score_) 
#

df_cv = pd.DataFrame(gcv.cv_results_)
