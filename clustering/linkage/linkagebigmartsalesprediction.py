#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:47:26 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram

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

#######################################
########### clustering and linkage

scaler = StandardScaler()
df_standard = scaler.fit_transform(grp_data)

merging = linkage(df_standard,method="average")
plt.figure(figsize=(9,6))
dendrogram(merging,labels=grp_data.index,leaf_rotation=90,leaf_font_size=8)
