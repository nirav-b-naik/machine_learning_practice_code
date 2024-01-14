#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:31:43 2022

@author: dai
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import joblib


#os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Wisconsin')

cancer= pd.read_csv('../Cases/Wisconsin/BreastCancer.csv',index_col=0)

d_cancer=pd.get_dummies(cancer,drop_first=True)

x=d_cancer.drop('Class_Malignant',axis=1)

y=d_cancer['Class_Malignant']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.3,
                                           stratify=y,
                                           random_state=2022)

clf=DecisionTreeClassifier(random_state=2022)

clf.fit(xtrain,ytrain)

y_pred=clf.predict_proba(xtest)[:,1]

print(roc_auc_score(ytest, y_pred)) #0.9263285024154588

##############################with grid search

kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

params={'max_depth':[3,5,7,None],
        'min_samples_split':[2,5,10,20],
        'min_samples_leaf':[1,5,10]}

gcv=GridSearchCV(clf, param_grid=params,cv=kfold,scoring='roc_auc')

gcv.fit(x,y)

print(gcv.best_params_)
# {'max_depth': 5, 'min_samples_leaf': 10, 'min_samples_split': 2}
print(gcv.best_score_)
# 0.9828184468961976

best_model = gcv.best_estimator_

y_pred=best_model.predict_proba(xtest)[:,1]

joblib.dump(best_model,"dtmodel01.model")

print("roc score with best model: ", roc_auc_score(ytest, y_pred)) #0.9263285024154588

plt.figure(figsize=(60,40))

plot_tree(best_model,feature_names=x.columns,
          class_names=['benign','malignant'],
          filled=True,fontsize=10)

print(best_model.feature_importances_)
# [0.00573143 0.83837819 0.07349875 0.00181225 0.         0.04917225
# 0.03140713 0.         0.        ]

ind=np.arange(9)
plt.figure()
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()
