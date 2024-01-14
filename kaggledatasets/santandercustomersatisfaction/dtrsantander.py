"""
NOTE: DO NOT RUN PROGRAM WITHOUT ASKING TO NIRAV.
1. PROGRAM IS VERY HEAVY AND WILL TAKE VERY LONG TIME AROUND 2-3 HOURS.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:06:12 2022

@author: dai
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

traindf=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/train.csv',index_col=0)

xtrain=traindf.drop('TARGET',axis=1)
ytrain=traindf['TARGET']

xtest= pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/test.csv',index_col=0)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

dtc=DecisionTreeClassifier(random_state=2022)

params={'max_depth': [None,2,5,10],
'min_samples_leaf': np.arange(1,11),
'min_samples_split': np.arange(2,11)}

####### DecisionTreeClassifier with cross val score

dtc.fit(xtrain,ytrain)

result1=cross_val_score(dtc,xtrain,ytrain,scoring='roc_auc',cv=kfold)

print(result1)
# [0.56875784 0.58193961 0.5711415  0.57451643 0.58064061]
print(result1.mean())
# 0.5753991947335767

y_pred_prob1=dtc.predict_proba(xtest)[:,1]
y_pred1=dtc.predict(xtest)

##############submission DecisionTreeClassifier with cross val score

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob1

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/santander_result_dtr_1.csv')



####### DecisionTreeClassifier with gridsearchcv

gcv2=GridSearchCV(dtc, param_grid=params,scoring='roc_auc',cv=kfold,verbose=1)

gcv2.fit(xtrain,ytrain)

print(gcv2.best_params_)
#{'max_depth': 5, 'min_samples_leaf': 8, 'min_samples_split': 2}

print(gcv2.best_score_)
#0.8147644013142289

best_model2=gcv2.best_estimator_

y_pred_prob2=best_model2.predict_proba(xtest)[:,1]
y_pred2=best_model2.predict(xtest)

##############submission DecisionTreeClassifier with gridsearchcv

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob2

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/santander_result_dtr_2.csv')



####### DecisionTreeClassifier with bagging

bag3=BaggingClassifier(base_estimator=dtc,random_state=2022)

bag3.fit(xtrain,ytrain)

y_pred_prob3=bag3.predict_proba(xtest)[:,1]
y_pred3=bag3.predict(xtest)


##############submission DecisionTreeClassifier with bagging

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob3

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/santander_result_dtr_3.csv')



####### DecisionTreeClassifier with bagging with gridsearchcv

bag4=BaggingClassifier(base_estimator=dtc,random_state=2022)

params={'base_estimator__max_depth': [None,3,5],
        'base_estimator__min_samples_leaf': np.arange(1,6),
        'base_estimator__min_samples_split': np.arange(2,7),
        'n_estimators': [10,15]}

gcv4=GridSearchCV(bag4, param_grid=params,scoring='roc_auc',cv=kfold,verbose=3)

gcv4.fit(xtrain,ytrain)
#starttime=6.52 pm
#stoptime=

print(gcv4.best_params_)
# {'base_estimator__max_depth': 5, 'base_estimator__min_samples_leaf': 4, 'base_estimator__min_samples_split': 2, 'n_estimators': 15}

print(gcv4.best_score_)
# 0.8264329356714339

best_model4=gcv4.best_estimator_

y_pred_prob4=best_model4.predict_proba(xtest)[:,1]
y_pred4=best_model4.predict(xtest)

##############submission DecisionTreeClassifier with bagging with gridsearchcv

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=y_pred_prob4

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/santandercustomersatisfaction/santander_result_dtr_4.csv')

