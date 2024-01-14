#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:54:37 2022

@author: dai
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import os
import matplotlib.pyplot as plt

b_concrete=pd.read_csv(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")

x = b_concrete.drop('Strength',axis=1)
y = b_concrete['Strength']

gmb = GradientBoostingRegressor()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
params = {"learning_rate":np.linspace(0.001,0.5,10),
          "max_depth":[2,3,4,5,6],
          "n_estimators":[50,100,150]}
gcv = GridSearchCV(gmb, param_grid=params,scoring="r2",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

df_cv = pd.DataFrame(gcv.cv_results_)

best_model = gcv.best_estimator_

imp=best_model.feature_importances_

i_sort=np.argsort(-imp)

sorted_imp=imp[i_sort]

sorted_col=x.columns[i_sort]


ind=np.arange(x.shape[1])

plt.bar(ind,sorted_imp)

plt.xticks(ind,(sorted_col),rotation=90)

plt.title('feature importance')

plt.xlabel('variables')
