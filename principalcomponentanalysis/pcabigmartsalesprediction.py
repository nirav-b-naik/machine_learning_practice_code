#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:06:37 2022

@author: dai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pca import pca

train = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/analyticalvidhya/bigmartsalesprediction/train_v9rqX0R.csv")

train.isna().sum()

train["Item_Weight"].fillna(train["Item_Weight"].mean(),inplace=True)

train["Outlet_Size"].fillna('Unknown',inplace=True)

train['Item_Fat_Content'].replace({'reg':'Regular',
                                'LF':'Low Fat',
                                'low fat':'Low Fat'},inplace=True)

avg_wt=train.groupby('Item_Type')['Item_Weight'].mean()
avg_vis=train.groupby('Item_Type')['Item_Visibility'].mean()
avg_mrp=train.groupby('Item_Type')['Item_MRP'].mean()
avg_sale=train.groupby('Item_Type')['Item_Outlet_Sales'].mean()

grp_data=pd.concat([avg_wt,avg_vis,avg_mrp,avg_sale],axis=1)

scaler = StandardScaler()
df_scl = scaler.fit_transform(grp_data)

PCA = PCA()
principalcomponent = PCA.fit_transform(df_scl)

print(PCA.explained_variance_)

print(np.sum(PCA.explained_variance_))

print(PCA.explained_variance_ratio_)

print(PCA.explained_variance_ratio_ * 100)

print(np.cumsum(PCA.explained_variance_ratio_*100))

pd_PC = pd.DataFrame(principalcomponent,columns=["PC1","PC2","PC3","PC4"],index = grp_data.index)

plt.scatter(pd_PC["PC1"],pd_PC["PC2"])
plt.show()

####################################################

pd_scl = pd.DataFrame(df_scl,columns=grp_data.columns,index = grp_data.index)

model = pca()

results = model.fit_transform(pd_scl)

plt.figure(figsize=(40,30))
fig, ax = model.biplot(label=True,legend=False)
