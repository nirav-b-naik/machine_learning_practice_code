#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:35:29 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing')

df = pd.read_csv("Boston.csv")
x = df.drop('medv',axis=1)
y = df['medv']

xgb=XGBRegressor(random_state=2022)

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

params = {'learning_rate':np.linspace(0.001, 0.5,10),
          'max_depth':[2,3,4,5,6],
          'n_estimators': [50,100,150]}

gcv = GridSearchCV(xgb, param_grid=params,cv=kfold,scoring='r2',verbose=10)

gcv.fit(x,y)

print(gcv.best_params_)
# {'learning_rate': 0.11188888888888888, 'max_depth': 3, 'n_estimators': 150}

print(gcv.best_score_)
# 0.8973817622783832

pd_cv=pd.DataFrame(gcv.cv_results_)

best_model=gcv.best_estimator_


imp=best_model.feature_importances_

i_sort=np.argsort(-imp)

sorted_imp=imp[i_sort]

sorted_col=x.columns[i_sort]


ind=np.arange(x.shape[1])

plt.bar(ind,sorted_imp)

plt.xticks(ind,(sorted_col),rotation=90)

plt.title('feature importance')

plt.xlabel('variables')

plt.show()






