#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:00:43 2022

@author: dai
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength")
df = pd.read_csv("Concrete_Data.csv")
df.columns

x = df.drop('Strength',axis=1)
y = df['Strength']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 random_state=2022,
                                                 test_size=0.3)

scaler = StandardScaler()
r2list=[]

for i in range(1,30):
    knn = KNeighborsRegressor(n_neighbors=i)
    pipe = Pipeline([('scaler_std',scaler),('knn_model',knn)])
    
    pipe.fit(x_train, y_train)
    
    y_pred = pipe.predict(x_test)
    
    #print(f'{i}: mean_absolute_error:',round(mean_absolute_error(y_test, y_pred),6),end="\t")
    #print('mean_squared_error:',round(mean_squared_error(y_test, y_pred),6),end="\t")
    #print('r2score:',round(r2_score(y_test, y_pred),6))
    r2list.append(round(r2_score(y_test, y_pred),6))
    
print(f'best k= {np.argmax(r2list)+1} with r2= {np.max(r2list)}')

#################Testset from other file#########################

test_insure=pd.read_csv("testConcrete.csv")

knn=KNeighborsRegressor(n_neighbors=5)
pipe= Pipeline([('scl_std',scaler),('knn_model',knn)])
pipe.fit(x_train,y_train)

#dum_insure = pd.get_dummies(test_insure,drop_first=True)

y_pred=pipe.predict(test_insure)

print(list(y_pred))

