#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:48:25 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import matplotlib.pyplot as plt
from xgboost import XGBRFClassifier

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/bank/bank.csv", sep=";")
df_dummy = pd.get_dummies(df,drop_first=True)

x = df_dummy.drop("y_yes",axis=1)
y = df_dummy["y_yes"]

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2022)

xgb=XGBRFClassifier(random_state=2022)

params = {"learning_rate":np.linspace(0.001,0.5,10),
          "max_depth":[2,3,4,5,6],
          "n_estimators":[50,100,150]}

gcv=GridSearchCV(xgb,param_grid=params,scoring='roc_auc',cv=kfold,verbose=10)

gcv.fit(x,y)

print(gcv.best_params_)
# {'learning_rate': 0.001, 'max_depth': 6, 'n_estimators': 150}

print(gcv.best_score_)
# 0.8846983058608059

best_model=gcv.best_estimator_


df_cv=pd.DataFrame(gcv.cv_results_)


imp=best_model.feature_importances_

i_sort=np.argsort(-imp)

sorted_imp=imp[i_sort]

sorted_col=x.columns[i_sort]


ind=np.arange(x.shape[1])

plt.bar(ind,sorted_imp)

plt.xticks(ind,(sorted_col),rotation=90)

plt.title('feature importance')

plt.xlabel('variables')






