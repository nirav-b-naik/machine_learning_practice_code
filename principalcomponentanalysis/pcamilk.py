#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:45:17 2022

@author: dai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pca import pca


df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/milk.csv",index_col=0)
scaler = StandardScaler()
df_scl = scaler.fit_transform(df)

PCA = PCA()
principalcomponent = PCA.fit_transform(df_scl)

print(PCA.explained_variance_)

print(np.sum(PCA.explained_variance_))

print(PCA.explained_variance_ratio_)

print(PCA.explained_variance_ratio_ * 100)

print(np.cumsum(PCA.explained_variance_ratio_*100))

pd_PC = pd.DataFrame(principalcomponent,columns=["PC1","PC2","PC3","PC4","PC5"],index = df.index)


plt.scatter(pd_PC["PC1"],pd_PC["PC2"])
plt.show()

################################################################################

pd_scl = pd.DataFrame(df_scl,columns=df.columns,index = df.index)

model = pca()

results = model.fit_transform(pd_scl)

fig, ax = model.biplot(label=True,legend=False)