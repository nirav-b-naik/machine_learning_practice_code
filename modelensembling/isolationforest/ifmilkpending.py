#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:56:01 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV,KFold
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
    
df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing/Boston.csv")
x = df.drop("medv",axis=1)
y = df["medv"]

clf = IsolationForest(contamination=0.05,random_state=2022)
clf.fit(x)

prediction = clf.predict(x)
series_outlier  = pd.Series(prediction,name="Outliers")

dt_outliers = pd.concat([df,series_outlier],axis=1)
outliers_df = dt_outliers[dt_outliers["Outliers"]==-1]
wothout_outlier_df = dt_outliers[dt_outliers["Outliers"]!=-1]
