#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:04:05 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets")
df = pd.read_csv("RidingMowers.csv")

dum_df=pd.get_dummies(df,drop_first=True)

sns.scatterplot(x='Income', y='Lot_Size', hue='Response', data=df)
plt.show()

x=dum_df.drop('Response_Not Bought',axis=1)
y=dum_df['Response_Not Bought']

x_train, x_test, y_train, y_test=train_test_split(x,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=2022,
                                                  stratify=y)

y.value_counts(normalize=True)*100
y_test.value_counts(normalize=True)*100
y_train.value_counts(normalize=True)*100

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

#ROC AUC

y_pred_prob=knn.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

y_pred=knn.predict(x_test)
print(accuracy_score(y_test, y_pred))

#best k=3
test_mowers=pd.read_csv('testMowers.csv')
predictions=knn.predict(test_mowers)
test_mowers['predictions']=predictions
