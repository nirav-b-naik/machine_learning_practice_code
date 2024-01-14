#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:07:55 2022

@author: dai
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rfm=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Recency Frequency Monetary/rfm_data_customer.csv',index_col=0)
rfm.drop("most_recent_visit",axis=1,inplace=True)
# scaling is compulsory
scaler=StandardScaler()
rfmscl=scaler.fit_transform(rfm)

######### Within Sum of Square

inertia=list()

for i in range(1,20):
    model=KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(rfmscl)
    val=model.inertia_
    inertia.append(val)

plt.plot(range(1,20),inertia,'-o')
plt.title('Screenplot')
plt.xlabel('No of Clusters, k')
plt.ylabel('Inertia')
plt.xticks(range(1,20))
plt.show()

model=KMeans(n_clusters=4,random_state=2022,verbose=10)
model.fit(rfmscl)
val=model.inertia_

######### silhouette_score

cluster=[x for x in range(2,10)]

lst = []
for i in cluster:
    model = KMeans(n_clusters=i,random_state=2022,verbose=10)
    model.fit(rfmscl)
    labels = model.predict(rfmscl)
    sil_score = silhouette_score(rfmscl, labels)    
    lst.append(sil_score)

i_max = np.argmax(lst)
k_max = cluster[i_max]

model = KMeans(n_clusters=3,random_state=2022,verbose=10)
model.fit(rfmscl)
labels = model.predict(rfmscl)

rfm['cluster']=labels

rfm.groupby('cluster').mean()

