#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:07:10 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

usa = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/USArrests.csv",index_col=0)
scaler = StandardScaler()
usa_scale = scaler.fit_transform(usa)

cluster=[x for x in range(2,45)]

lst = []
for i in cluster:
    model = KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(usa_scale)
    labels = model.predict(usa_scale)
    sil_score = silhouette_score(usa_scale, labels)
    lst.append(sil_score)

i_max = np.argmax(lst)
k_max = cluster[i_max]

print(f"{k_max}-------> {max(lst)}")

model = KMeans(n_clusters=2,random_state=2022,verbose=10)
model.fit(usa_scale)
labels = model.predict(usa_scale)

usa['cluster']=labels

usa.groupby('cluster').mean()
