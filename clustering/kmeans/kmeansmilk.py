#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:20:59 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

milk=pd.read_csv('../../Datasets/milk.csv',index_col=0)

# scaling is compulsory
scaler=StandardScaler()
milkscale=scaler.fit_transform(milk)

model=KMeans(n_clusters=3,random_state=2022,verbose=10)
model.fit(milkscale)

print(model.cluster_centers_)

labels=model.predict(milkscale)
milk['cluster']=labels

milk.sort_values('cluster',inplace=True)

print(model.inertia_)


######### Within Sum of Square

inertia=list()

for i in range(1,20):
    model=KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(milkscale)
    val=model.inertia_
    inertia.append(val)

plt.plot(range(1,20),inertia,'-o')
plt.title('Screenplot')
plt.xlabel('No of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(range(1,20))
plt.show()

######### silhouette_score

print(silhouette_score(milkscale,labels))

silscore=list()

for i in range(2,15):
    model=KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(milkscale)
    labels=model.predict(milkscale)
    s_score=silhouette_score(milkscale,labels)
    silscore.append(s_score)

print(f' max sil_score: {np.max(silscore)}  N of cluster: {np.argmax(silscore)+2}')

plt.plot(range(2,15),silscore,'-o')
plt.title('Screenplot')
plt.xlabel('No of Clusters, k')
plt.ylabel('silscore')
plt.xticks(range(2,15))
plt.show()




