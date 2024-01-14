#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:23:26 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pca import pca

cars=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Cars93.csv')

cars_small=cars[cars['Type']=='Small']

carprice=cars_small.groupby('Manufacturer')['Price'].mean()
carmpgc=cars_small.groupby('Manufacturer')['MPG.city'].mean()
carmpgh=cars_small.groupby('Manufacturer')['MPG.highway'].mean()
careng=cars_small.groupby('Manufacturer')['EngineSize'].mean()
carhp=cars_small.groupby('Manufacturer')['Horsepower'].mean()
carcap=cars_small.groupby('Manufacturer')['Fuel.tank.capacity'].mean()
carrev=cars_small.groupby('Manufacturer')['Rev.per.mile'].mean()

grp_data=pd.concat([carprice,carmpgc,carmpgh,careng,carhp,carcap,carrev],axis=1)

scaler = StandardScaler()
df_standard = scaler.fit_transform(grp_data)

PCA=PCA()

principalcomponent=PCA.fit_transform(df_standard)

print(PCA.explained_variance_)

print(np.sum(PCA.explained_variance_))

print(PCA.explained_variance_ratio_)

print(PCA.explained_variance_ratio_*100)

print(np.cumsum(PCA.explained_variance_ratio_*100))

pd_PC=pd.DataFrame(principalcomponent,columns=["PC1","PC2","PC3","PC4","PC5","PC6","PC7"],index=grp_data.index)

plt.scatter(pd_PC["PC1"],pd_PC["PC2"])

plt.show()


###############################################################


pd_scl=pd.DataFrame(df_standard,columns=grp_data.columns,index=grp_data.index)

model=pca()

results = model.fit_transform(pd_scl)

plt.figure(figsize=(40,30))
fig, ax = model.biplot(label=True,legend=False)