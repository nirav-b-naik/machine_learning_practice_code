#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:45:45 2022

@author: dai
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline

train=pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/train.csv",index_col=0)
x = train.drop('TARGET',axis=1)
y = train["TARGET"]
xtest=pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/test.csv",index_col=0)

sgd = SGDClassifier(loss='log_loss',random_state=2022)
sgd.fit(x,y)

y_pred_prob = sgd.predict_proba(xtest)[:,1]
#print(roc_auc_score(ytest, y_pred_prob))

###################### KFold CV ############################################

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
sgd1 = SGDClassifier(loss='log_loss',random_state=2022)
result = cross_val_score(sgd1,x,y,scoring='roc_auc',cv=kfold,verbose=1)
print(result.mean())

sgd1.fit(x,y)

y_pred_prob = sgd1.predict_proba(x)[:,1]
print(roc_auc_score(y, y_pred_prob))

######################### GridSearchCv ############################################
scaler = MinMaxScaler()
pipe = Pipeline([('scl',scaler),('SGD',sgd)])
params = {'SGD__learning_rate':['constant','optimal','invscaling','adaptive'],'SGD__eta0':np.linspace(0.001, 0.7,5)}

gcv = GridSearchCV(pipe,param_grid=params,cv=kfold,scoring='roc_auc',verbose=1)

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

sgd = SGDClassifier(learning_rate='adaptive',eta0=0.7,loss='log_loss',random_state=2022)

sgd.fit(x,y)

ypred=sgd.predict_proba(xtest)[:,1]

####### Submition for SGDClassifier  ##############

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sample_submission.csv',index_col=0)

submit['TARGET']=ypred

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kagglecompetition/santandercustomersatisfaction/sub_result_gdclass_new.csv')
