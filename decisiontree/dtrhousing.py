#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:20:57 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,GridSearchCV,cross_val_score,KFold
from sklearn.tree import DecisionTreeRegressor , plot_tree
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets')
    
house= pd.read_csv('Housing.csv')

dummy_house=pd.get_dummies(house,drop_first=True)

x=dummy_house.drop('price',axis=1)

y=dummy_house['price']

##############################with linear regresion

kfold=KFold(n_splits=5, shuffle=True, random_state=2022)

lr=LinearRegression()

result=cross_val_score(lr,x,y,scoring='r2',cv=kfold)

print(result)
print(result.mean())

##############################with decisiontreeregressor

kfold=KFold(n_splits=5, shuffle=True, random_state=2022)

clf=DecisionTreeRegressor(random_state=2022)

params={'max_depth':[None,1,2,3,4,5,6,7,8,9,10,11],
        'min_samples_split':np.arange(2,11),
        'min_samples_leaf':np.arange(1,11)}

gcv=GridSearchCV(clf, param_grid=params,cv=kfold,scoring='r2')

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

plt.figure(figsize=(100,50))
plot_tree(best_model,feature_names=x.columns,
          filled=True,fontsize=10)

print(best_model.feature_importances_)
ind=np.arange(x.shape[1])
plt.figure()
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()