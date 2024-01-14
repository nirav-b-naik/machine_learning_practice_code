#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:01:44 2022

@author: dai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/ottogroup")

train = pd.read_csv("train.csv",index_col=0)
test = pd.read_csv("test.csv",index_col=0)

x = train.drop("target",axis=1)
y =train["target"]

le = LabelEncoder()
le_y  = le.fit_transform(y)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

dtc = DecisionTreeClassifier(random_state=2022)

####### DecisionTreeClassifier with cross_val_score

dtc.fit(x, le_y)

result1 = cross_val_score(dtc,x,le_y,cv=kfold,scoring="neg_log_loss")

print(result1)
# [ -9.9156652   -9.70914698 -10.08869398  -9.90251141 -10.10067327]

print(result1.mean())
# -9.94333816877214

y_pred_prob1 = dtc.predict_proba(test)

pd_pred_prob1 = pd.DataFrame(y_pred_prob1,columns=list(le.classes_))

####### Submit DecisionTreeClassifier with cross_val_score

submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob1],axis=1)

submission.to_csv("submit_dtc_1.csv",index=False)



##########################################################################
####### DecisionTreeClassifier with gridsearchcv

params2={'max_depth':[3,5,7,None],
        'min_samples_split':[2,5,10,20],
        'min_samples_leaf':[1,5,10]}

gcv2=GridSearchCV(dtc,param_grid=params2,cv=kfold,scoring='neg_log_loss',verbose=5)

gcv2.fit(x,y)

print(gcv2.best_params_)
# {'max_depth': 7, 'min_samples_leaf': 5, 'min_samples_split': 20}

print(gcv2.best_score_)
# -1.2162815231424409

best_model2 = gcv2.best_estimator_

y_pred_prob2 = best_model2.predict_proba(test)

pd_pred_prob2 = pd.DataFrame(y_pred_prob2,columns=list(le.classes_))

####### Submit DecisionTreeClassifier with gridsearchcv

submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob2],axis=1)

submission.to_csv("submit_dtc_2.csv",index=False)



##########################################################################
####### DecisionTreeClassifier with gridsearchcv with bagging

bag3=BaggingClassifier(base_estimator=dtc,random_state=2022)

params3={'base_estimator__max_depth': [None,3,8,10],
'base_estimator__min_samples_leaf': np.arange(1,11),
'base_estimator__min_samples_split': np.arange(2,11),
'n_estimators': [10,20,50]}

gcv3=GridSearchCV(bag3,param_grid=params3,cv=kfold,scoring='neg_log_loss',verbose=3)

gcv3.fit(x,y)

print(gcv3.best_params_)
# {'base_estimator__max_depth': None, 'base_estimator__min_samples_leaf': 5, 'base_estimator__min_samples_split': 2, 'n_estimators': 50}

print(gcv3.best_score_)
#-0.627858907085273

best_model3 = gcv3.best_estimator_

y_pred_prob3 = best_model3.predict_proba(test)

pd_pred_prob3 = pd.DataFrame(y_pred_prob3,columns=list(le.classes_))

####### Submit DecisionTreeClassifier with gridsearchcv

submit = pd.read_csv("sampleSubmission.csv")

submission = pd.concat([submit['id'],pd_pred_prob3],axis=1)

submission.to_csv("submit_dtc_3.csv",index=False)
