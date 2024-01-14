#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:12:43 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

nutrient = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/nutrient.csv",index_col=0)

scaler = StandardScaler()
nutrient_scl = scaler.fit_transform(nutrient)

cluster_db = DBSCAN(eps=1,min_samples=4)
cluster_db.fit(nutrient_scl)
print(cluster_db.labels_)

labels = cluster_db.labels_
print(silhouette_score(nutrient_scl, labels))

eps_range = [x/10 for x in range(2,20)]
mp_range = [x for x in range(1,20)]
cnt = 0

lst = []

for i in eps_range:
    for j in mp_range:
        cluster_db = DBSCAN(eps=i,min_samples=j)
        cluster_db.fit(nutrient_scl)
        labels = cluster_db.labels_
        if len(set(labels)) > 2:
            cnt = cnt + 1            
            sil_sc = silhouette_score(nutrient_scl, labels)
            lst.append([cnt,i,j,sil_sc]) 
            print(cnt,'->',i,'->',j,'->',sil_sc)

pa=pd.DataFrame(lst,columns=['sr','i','j','sil'])

pa[pa['sil']==pa['sil'].max()]

cluster_db = DBSCAN(eps=1,min_samples=2)

cluster_db.fit(nutrient_scl)

labels = cluster_db.labels_

nutrient['cluster']=labels



            
            
            
            
            
            
            
            