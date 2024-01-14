#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:36:51 2022

@author: dai
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Image Segmentation')
df=pd.read_csv('Image_Segmention.csv')

x = df.drop('Class',axis=1)
y = df['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
kfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

#####################KNN###########################
knn= KNeighborsClassifier()
scaler = StandardScaler()

pipe=Pipeline([('scl_std',scaler),('knn_model',knn)])
params = {'knn_model__n_neighbors':np.arange(1,30)}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(x,le_y)
print(gcv.best_params_)
print(gcv.best_score_)
gcv_est = gcv.best_estimator_
df1 = pd.DataFrame(gcv.cv_results_)

########################GaussianNB########################

gnb=GaussianNB()

pipe = Pipeline([('scl_std',scaler),('gnb_model',gnb)])

result = cross_val_score(pipe, x,y,scoring='neg_log_loss',cv=kfold)

print(result)
print(result.mean())


###########################Testfile####################

tst_img = pd.read_csv("tst_img.csv")
best_model = gcv.best_estimator_

y_pred = best_model.predict(tst_img)
print(le.inverse_transform(y_pred))












