#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:36:14 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score

image_seg = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Image Segmentation/Image_Segmention.csv")
x = image_seg.drop('Class',axis=1)
y = image_seg['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params={'penalty': ['l1','l2','elasticnet','none'],
        'multi_class' : ['ovr','multinomial']}
logreg = LogisticRegression()
gcv = GridSearchCV(logreg, param_grid=params,scoring='neg_log_loss',cv=kfold)
gcv.fit(x,le_y)

print(gcv.best_params_)
print(gcv.best_score_)

gcv_df = pd.DataFrame(gcv.cv_results_)


################################## roc_auc_score ovo,ovr ############
xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,random_state=2022)

logreg=LogisticRegression()
logreg.fit(xtrain,ytrain)
y_pred_prob=logreg.predict_proba(xtest)

#AUC_ROC_SCORE_RAISE
#print(roc_auc_score(ytest, y_pred_prob))

#AUC_ROC_SCORE_OVR
print(roc_auc_score(ytest, y_pred_prob,multi_class='ovr'))

#AUC_ROC_SCORE_OVO
print(roc_auc_score(ytest, y_pred_prob,multi_class='ovo'))





