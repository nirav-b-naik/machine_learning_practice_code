#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:44:01 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import matplotlib.pyplot as plt
from xgboost import XGBRFClassifier

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Bankruptcy/Bankruptcy.csv",index_col=0)
x = df.drop(['D', 'YR'],axis=1)
y = df['D']

kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=2022)

xgb=XGBRFClassifier(random_state=2022)

params = {"learning_rate":np.linspace(0.001,0.5,10),
          "max_depth":[2,3,4,5,6],
          "n_estimators":[50,100,150]}

gcv=GridSearchCV(xgb,param_grid=params,scoring='roc_auc',cv=kfold,verbose=10)

gcv.fit(x,y)

print(gcv.best_params_)
# {'learning_rate': 0.05644444444444444, 'max_depth': 2, 'n_estimators': 150}

print(gcv.best_score_)
# 0.9227387996618767

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






