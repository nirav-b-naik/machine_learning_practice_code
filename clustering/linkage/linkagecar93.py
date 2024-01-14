#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:03:20 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram

cars=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Cars93.csv')

cars_small=cars[cars['Type']=='Small']

carprice=cars_small.groupby('Manufacturer')['Max.Price'].mean()
carmpg=cars_small.groupby('Manufacturer')['MPG.city'].mean()
carpower=cars_small.groupby('Manufacturer')['Horsepower'].mean()
carbase=cars_small.groupby('Manufacturer')['Wheelbase'].mean()
carrpm=cars_small.groupby('Manufacturer')['RPM'].mean()

grp_data=pd.concat([carprice,carmpg,carpower,carbase,carrpm],axis=1)

#######################################
########### clustering and linkage

scaler = StandardScaler()
df_standard = scaler.fit_transform(grp_data)

merging = linkage(df_standard,method="average")
plt.figure(figsize=(9,6))
dendrogram(merging,labels=grp_data.index,leaf_rotation=90,leaf_font_size=8)