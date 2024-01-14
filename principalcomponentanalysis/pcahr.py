#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:16:22 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics/HR_comma_sep.csv")
df_dummy = pd.get_dummies(df,drop_first=True)
x = df_dummy.drop("left",axis=1)
y = df_dummy["left"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,stratify=y,random_state=2022)
pca = PCA(random_state=2022)

scaler = StandardScaler()
xtrain_scl = scaler.fit_transform(xtrain)

principle_comp = pca.fit_transform(xtrain_scl)

print(np.cumsum(pca.explained_variance_ratio_)*100)

######################################################

svc = SVC(probability=True,random_state=2022)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
pipe = Pipeline([("Scl",scaler),("PCA",pca),("SVC",svc)])
params = {'PCA__n_components': [0.85,0.90,0.95],
          'SVC__C': [0.5,1,1.5],
          'SVC__gamma': ['scale','auto']}

gcv = GridSearchCV(pipe, param_grid=params,scoring="roc_auc",cv=kfold,verbose=10)
gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

###########################

plt.figure()
plt.plot(np.arange(1,x.shape[1]+1),(np.cumsum(pca.explained_variance_ratio_)*100))
plt.show()


