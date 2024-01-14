#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:20:27 2022

@author: dai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram

################################### nutrient
df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/nutrient.csv",index_col=0)
scaler = StandardScaler()
df_standard = scaler.fit_transform(df)

merging = linkage(df_standard,method="average")
dendrogram(merging,labels=df.index,leaf_rotation=90,leaf_font_size=10)

##################################### US-arrest
df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/USArrests.csv",index_col=0)
scaler = StandardScaler()
df_standard = scaler.fit_transform(df)

merging = linkage(df_standard,method="complete")
plt.figure(figsize=(12,8))
dendrogram(merging,labels=df.index,leaf_rotation=90,leaf_font_size=8)