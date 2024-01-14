#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:40:06 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV,KFold
from xgboost import XGBRegressor

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")
x = df.drop(["Strength"],axis=1)
y = df["Strength"]

xgb=XGBRegressor(random_state=2022)

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)

params = {'learning_rate':[0.001, 0.1,0.4],
          'max_depth':[2,4,6],
          'n_estimators': [50,100]}

gcv = GridSearchCV(xgb, param_grid=params,cv=kfold,scoring='r2',verbose=10)

gcv.fit(x,y)

print(gcv.best_params_)
# {'learning_rate': 0.4, 'max_depth': 4, 'n_estimators': 100}

print(gcv.best_score_)
# 0.933921234504828


############################################## Kmeans

scaler = StandardScaler()
x_scl = scaler.fit_transform(x)
lst = []
cluster = [x for x in range(2,10)]
for i in cluster:
    model = KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(x_scl)
    labels = model.predict(x_scl)
    sil_score = silhouette_score(x_scl, labels)   
    lst.append(sil_score)
    
i_max = np.argmax(lst)
k_max = cluster[i_max]

print(k_max)

model = KMeans(n_clusters=k_max,random_state=2022,verbose=10)
model.fit(x_scl)
labels = model.predict(x_scl)
sil_score = silhouette_score(x_scl, labels)   
# 0.28867112224826236

x["c"] = labels
x["c"]=x["c"].astype('category')
x_aug=pd.get_dummies(x)


gcv1 = GridSearchCV(xgb, param_grid=params,cv=kfold,scoring='r2',verbose=10)

gcv1.fit(x_aug,y)

print(gcv1.best_params_)
# {'learning_rate': 0.4, 'max_depth': 4, 'n_estimators': 100}

print(gcv1.best_score_)
# 0.9343867606122626