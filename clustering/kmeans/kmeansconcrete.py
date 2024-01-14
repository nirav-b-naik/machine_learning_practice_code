#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:12:00 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv")
x = df.drop(["Strength"],axis=1)
y = df["Strength"]

x_train, x_test, y_train, y_test=train_test_split(x,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=2022)

rfc = RandomForestRegressor(random_state=2022,verbose=10)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
print(r2_score(y_test, y_pred))  
# 0.9093110266698041

############################################## K-Means
scaler = StandardScaler()
xtrain_scl = scaler.fit_transform(x_train)
lst = []
cluster = [x for x in range(2,10)]
for i in cluster:
    model = KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(xtrain_scl)
    labels = model.predict(xtrain_scl)
    sil_score = silhouette_score(xtrain_scl, labels)   
    lst.append(sil_score)
    
i_max = np.argmax(lst)
k_max = cluster[i_max]

print(k_max)

model = KMeans(n_clusters=k_max,random_state=2022,verbose=10)
model.fit(xtrain_scl)
labels = model.predict(xtrain_scl)
sil_score = silhouette_score(xtrain_scl, labels)   
# 0.2886785627685946

x_train["c"] = labels
x_train["c"]=x_train["c"].astype('category')
x_trn_aug=pd.get_dummies(x_train)

xtest_scl = scaler.transform(x_test)

labels = model.predict(xtest_scl)
sil_score = silhouette_score(xtest_scl, labels)  
# 0.2787541402841135

x_test["c"] = labels
x_test["c"]=x_train["c"].astype('category')
x_test_aug=pd.get_dummies(x_test)

rf = RandomForestRegressor(random_state=2022)
rf.fit(x_trn_aug, y_train)

y_pred= rf.predict(x_test_aug)
print(r2_score(y_test, y_pred))
# 0.9035573561579231
