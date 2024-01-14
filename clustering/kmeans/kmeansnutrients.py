#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:24:03 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nutrients = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/nutrient.csv",index_col=0)
scaler = StandardScaler()
nutrients_scale = scaler.fit_transform(nutrients)

cluster=[x for x in range(2,10)]

lst = []
for i in cluster:
    model = KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(nutrients_scale)
    labels = model.predict(nutrients_scale)
    sil_score = silhouette_score(nutrients_scale, labels)
    lst.append(sil_score)

i_max = np.argmax(lst)
k_max = cluster[i_max]

print(f"{k_max}-------> {max(lst)}")

model = KMeans(n_clusters=4,random_state=2022,verbose=10)
model.fit(nutrients_scale)
labels = model.predict(nutrients_scale)

nutrients['cluster']=labels

nutrients.groupby('cluster').mean()
