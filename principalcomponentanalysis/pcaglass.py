#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:43:43 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from pca import pca

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Glass Identification/Glass.csv")
x = df.drop("Type",axis=1)

y = df["Type"]

le=LabelEncoder()

le_y=le.fit_transform(y)

xtrain,xtest,ytrain,ytest = train_test_split(x,le_y,test_size=0.3,stratify=le_y,random_state=2022)

######################################################

PCA = PCA(random_state=2022)

scaler = StandardScaler()
xtrain_scl = scaler.fit_transform(xtrain)

principle_comp = PCA.fit_transform(xtrain_scl)

print(np.cumsum(PCA.explained_variance_ratio_)*100)

######################################################

pd_PC= pd.DataFrame(principle_comp,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9'])

plt.figure(figsize=(300,200))

sns.scatterplot(pd_PC,x="pc1",y="pc2",hue=y,palette="deep")

plt.show()

######################################################

model=pca()

results = model.fit_transform(x)

plt.figure(figsize=(40,30))
fig, ax = model.biplot(label=True,legend=False)

######################################################

svc = SVC(probability=True,random_state=2022)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
pipe = Pipeline([("Scl",scaler),("PCA",PCA),("SVC",svc)])
params = {'PCA__n_components': [0.85,0.90,0.95],
          'SVC__C': [0.5,1,1.5],
          'SVC__gamma': ['scale','auto']}

gcv = GridSearchCV(pipe, param_grid=params,scoring="roc_auc_ovo",cv=kfold,verbose=10)
gcv.fit(x,le_y)

print(gcv.best_params_)
# {'PCA__n_components': 0.85, 'SVC__C': 1.5, 'SVC__gamma': 'auto'} - neg_log_loss
# {'PCA__n_components': 0.9, 'SVC__C': 1.5, 'SVC__gamma': 'auto'} - roc_auc_ovr
# {'PCA__n_components': 0.95, 'SVC__C': 1.5, 'SVC__gamma': 'scale'} - roc_auc_ovo

print(gcv.best_score_)
# -0.8419529990884712 - neg_log_loss
# 8839032356883326 - roc_auc_ovr
# 0.9062070105820107 - roc_auc_ovo

######################################################

plt.figure()
plt.plot(np.arange(1,x.shape[1]+1),(np.cumsum(PCA.explained_variance_ratio_)*100))
plt.show()


