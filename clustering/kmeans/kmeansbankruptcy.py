#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:38:04 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../Cases/Bankruptcy/Bankruptcy.csv",index_col=0)
dum_df = pd.get_dummies(df,drop_first=True)
x = dum_df.drop(["YR","D"],axis=1)
y = dum_df["D"]

x_train, x_test, y_train, y_test=train_test_split(x,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=2022,
                                                  stratify=y)

rfc = RandomForestClassifier(random_state=2022)
rfc.fit(x_train,y_train)

y_pred = rfc.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred))
# 0.94125

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
# 0.2154117826336732

x_train["c"] = labels
x_train["c"]=x_train["c"].astype('category')
x_trn_aug=pd.get_dummies(x_train)

xtest_scl = scaler.transform(x_test)

model.fit(xtest_scl)
labels = model.predict(xtest_scl)
sil_score = silhouette_score(xtest_scl, labels)  
# 0.23469062445214997

x_test["c"] = labels
x_test["c"]=x_test["c"].astype('category')
x_test_aug=pd.get_dummies(x_test)

rf = RandomForestClassifier(random_state=2022)
rf.fit(x_trn_aug, y_train)

y_pred_prob = rf.predict(x_test_aug)
print(roc_auc_score(y_test, y_pred_prob))
#0.93625
