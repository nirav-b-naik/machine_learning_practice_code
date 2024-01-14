#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:01:12 2022

@author: dai
"""

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


os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin")
df = pd.read_csv("BreastCancer.csv")
df=df.drop('Code',axis=1)
dum_df=pd.get_dummies(df,drop_first=True)


x=dum_df.drop('Class_Malignant',axis=1)
y=dum_df['Class_Malignant']

x_train, x_test, y_train, y_test=train_test_split(x,
                                                  y,
                                                  test_size=0.3,
                                                  random_state=2022,
                                                  stratify=y)

from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier

for i in range(1,100,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    
    #ROC AUC
    y_pred_prob=knn.predict_proba(x_test)[:,1]
    
    
    y_pred=knn.predict(x_test)  
    print(f"{i} roc_auc_score: ",round(roc_auc_score(y_test, y_pred_prob),5),end="\t\t")
    print(f"{i} accuracy_score: ",round(accuracy_score(y_test, y_pred),5))
'''
#best k=3

test_mowers=pd.read_csv('testMowers.csv')
predictions=knn.predict(test_mowers)
test_mowers['predictions']=predictions'''
