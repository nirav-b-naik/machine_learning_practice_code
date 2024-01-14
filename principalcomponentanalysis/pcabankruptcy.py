#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:26:03 2022

@author: dai
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

bank=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Bankruptcy/Bankruptcy.csv',index_col=0)

x=bank.drop('D',axis=1)
y=bank['D']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,
                                           test_size=0.3,random_state=2022)

svm=SVC(probability=True, random_state=2022)

scaler=StandardScaler()
scl_xtrain=scaler.fit_transform(xtrain)

pca=PCA()
princ_xtrain=pca.fit_transform(scl_xtrain)

#cumsum
print(np.cumsum(pca.explained_variance_ratio_*100))

prn_trn=princ_xtrain[:,:7]
svm.fit(prn_trn,ytrain)

#############test

scl_xtest=scaler.transform(xtest)
princ_xtest=pca.transform(scl_xtest)
prn_tst=princ_xtest[:,:7]

y_pred=svm.predict_log_proba(prn_tst)[:,1]

print(roc_auc_score(ytest, y_pred))
# 0.855


########################################################################
########################################################################

pca=PCA(n_components=0.90,random_state=2022)
princ_xtrain=pca.fit_transform(scl_xtrain)

#cumsum
print(np.cumsum(pca.explained_variance_ratio_*100))

prn_trn=princ_xtrain
svm.fit(prn_trn,ytrain)

#############test

scl_xtest=scaler.transform(xtest)
princ_xtest=pca.transform(scl_xtest)
prn_tst=princ_xtest

y_pred=svm.predict_log_proba(prn_tst)[:,1]

print(roc_auc_score(ytest, y_pred))
# 0.8724999999999999


########################################################################
########################################################################
################with pipe

svm=SVC(probability=True, random_state=2022)

scaler=StandardScaler()

pca=PCA(random_state=2022)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

pipe=Pipeline([('scl',scaler),('pca',pca),('svc',svm)])

params={'pca__n_components': np.arange(0.50,1,0.05),
        'svc__C':np.linspace(0.001,5,10),
          'svc__gamma': np.linspace(0.001,5,10)}

gcv=GridSearchCV(pipe, param_grid=params,scoring='roc_auc',cv=kfold,verbose=10)

gcv.fit(x,y)

print(gcv.best_params_)
# {'pca__n_components': 0.85, 'svc__C': 0.5564444444444444, 'svc__gamma': 0.001}
# {'pca__n_components': 0.7000000000000002, 'svc__C': 3.3336666666666663, 'svc__gamma': 0.001}

print(gcv.best_score_)
# 0.8783601014370246
# 0.8827557058326289

best_model=gcv.best_estimator_

y_pred1=best_model.predict_proba(x)[:,1]

print(roc_auc_score(y, y_pred1))
#0.8983011937557392

###########################

plt.figure()
plt.plot(np.arange(1,xtrain.shape[1]+1),(np.cumsum(pca.explained_variance_ratio_)*100))
plt.show()
