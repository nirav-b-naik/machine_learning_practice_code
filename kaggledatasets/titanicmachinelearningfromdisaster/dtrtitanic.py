#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:43:43 2022

@author: dai
"""

"""
NOTE: DO NOT RUN PROGRAM WITHOUT ASKING TO NIRAV.
1. PROGRAM IS VERY HEAVY AND WILL TAKE VERY LONG TIME 
AROUND 20-24 HOURS.
"""


import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import os

start_time = time.time()
os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/titanicmachinelearningfromdisaster')

################train dataset

traindf=pd.read_csv('train.csv',index_col=0)

xtrain_raw=traindf.drop(['Survived','Name','Cabin', 'Embarked','Ticket'],axis=1)

xtrain_raw['SibSp']=xtrain_raw['SibSp'].astype('category')
xtrain_raw['Parch']=xtrain_raw['Parch'].astype('category')

xtrain1=pd.get_dummies(xtrain_raw,drop_first=True)

ytrain=traindf['Survived']

################test dataset

testdf=pd.read_csv('test.csv',index_col=0)

xtest_raw=testdf.drop(['Name','Cabin', 'Embarked','Ticket'],axis=1)

xtest_raw['SibSp']=xtest_raw['SibSp'].astype('category')
xtest_raw['Parch']=xtest_raw['Parch'].astype('category')

xtest1=pd.get_dummies(xtest_raw,drop_first=True)

xtest1=xtest1.drop('Parch_9',axis=1)
print(f'time = {time.time()-start_time}')
#########################

imputer=SimpleImputer()
xtrain=imputer.fit_transform(xtrain1)
xtest=imputer.fit_transform(xtest1)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)


##############################################################################
##############################################################################
####### RandomForestClassifier with gridsearchcv

rfc=RandomForestClassifier(random_state=2022)

pipe1=Pipeline([('imp',imputer),('rfc',rfc)])

param1={'imp__strategy': ['mean','median'],
        'rfc__max_features': [2,3,4],
        'rfc__min_samples_leaf': np.arange(1,11,1),
        'rfc__min_samples_split': np.arange(2,12,1)}

gcv1=GridSearchCV(pipe1,param_grid=param1,scoring='accuracy',cv=kfold,verbose=10)

gcv1.fit(xtrain,ytrain)

print(gcv1.best_params_)
# {'imp__strategy': 'mean', 'rfc__max_features': 3, 'rfc__min_samples_leaf': 2, 'rfc__min_samples_split': 10}

print(gcv1.best_score_)
# 0.8361496453455528

best_model1=gcv1.best_estimator_

y_pred_prob1=best_model1.predict_proba(xtest)[:,1]

y_pred_train=best_model1.predict(xtrain)

print(accuracy_score(ytrain, y_pred_train))
#0.8843995510662177

y_pred1=best_model1.predict(xtest)


##############submission RandomForestClassifier with gridsearchcv

submit=pd.read_csv('gender_submission.csv',index_col=0)

submit['Survived']=y_pred_prob1

submit.to_csv('titanic_submission_01.csv')


##############################################################################
##############################################################################
####### KNeighborsClassifier with gridsearchcv

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

pipe2=Pipeline([('imp',imputer),('knn',knn)])

param2={'knn__n_neighbors':np.arange(1,500,2)}

gcv2=GridSearchCV(pipe2,param_grid=param2,scoring='accuracy',cv=kfold,verbose=10)

gcv2.fit(xtrain,ytrain)

print(gcv2.best_params_)
# {'knn__n_neighbors': 5}

print(gcv2.best_score_)
# 0.7171866172870504

best_model2=gcv2.best_estimator_

y_pred_prob2=best_model2.predict_proba(xtest)[:,1]

y_pred_train=best_model2.predict(xtrain)

print(accuracy_score(ytrain, y_pred_train))
# 0.8148148148148148

y_pred2=best_model2.predict(xtest)


##############submission KNeighborsClassifier with gridsearchcv

submit=pd.read_csv('gender_submission.csv',index_col=0)

submit['Survived']=y_pred2

submit.to_csv('titanic_submission_02.csv')


##############################################################################
##############################################################################
####### LogisticRegression with gridsearchcv

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression(random_state=2022)

pipe3=Pipeline([('imp',imputer),('logreg',logreg)])

param3={'imp__strategy':['mean','median'],
        'logreg__l1_ratio': np.linspace(0.1,1,10),
        'logreg__max_iter': [50,100,150,200,500,1000]}

gcv3=GridSearchCV(pipe3,param_grid=param3,scoring='accuracy',cv=kfold,verbose=10)

gcv3.fit(xtrain,ytrain)

print(gcv3.best_params_)
# {'imp__strategy': 'mean', 'logreg__l1_ratio': 0.1, 'logreg__max_iter': 200}

print(gcv3.best_score_)
# 0.8024480572468772

best_model3=gcv3.best_estimator_

y_pred_prob3=best_model3.predict_proba(xtest)[:,1]

y_pred_train=best_model3.predict(xtrain)

print(accuracy_score(ytrain, y_pred_train))
# 0.813692480359147

y_pred3=best_model3.predict(xtest)


##############submission LogisticRegression with gridsearchcv

submit=pd.read_csv('gender_submission.csv',index_col=0)

submit['Survived']=y_pred3

submit.to_csv('titanic_submission_03.csv')

##############################################################################
##############################################################################
####### VotingClassifier with gridsearchcv part 4

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

lda= LinearDiscriminantAnalysis()
svm_l= SVC(kernel='linear',probability=True,random_state=2022)
svm_c= SVC(kernel='rbf',probability=True,random_state=2022)
dtc= DecisionTreeClassifier(random_state=2022)

models4=[('logreg',logreg),('svm_l',svm_l),('svm_c',svm_c),('lda',lda),('dtc',dtc)]

voting4 = VotingClassifier(models4,voting='soft')

params4={'logreg__max_iter': [100,150,200,500,1000],
         'dtc__max_depth': [None,3,5,10],
         'svm_l__C':np.linspace(0.001,1,10),
         'svm_c__C':np.linspace(0.001,1,10)}

gcv4=GridSearchCV(voting4, param_grid=params4,scoring='accuracy',cv=kfold,verbose=10)

start_time4 = time.time()

gcv4.fit(xtrain,ytrain)

print(f'time4 = {time.time()-start_time4}')

print(gcv4.best_params_)
# 

print(gcv4.best_score_)
# 

best_model4=gcv4.best_estimator_

y_pred_prob4=best_model4.predict_proba(xtest)[:,1]

y_pred_train=best_model4.predict(xtrain)

print(accuracy_score(ytrain, y_pred_train))
# 

y_pred4=best_model4.predict(xtest)


##############submission VotingClassifier with gridsearchcv

submit=pd.read_csv('gender_submission.csv',index_col=0)

submit['Survived']=y_pred4

submit.to_csv('titanic_submission_04.csv')


##############################################################################
##############################################################################
####### VotingClassifier with cross_val_score

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

lda= LinearDiscriminantAnalysis()
svm_l= SVC(kernel='linear',probability=True,random_state=2022)
svm_c= SVC(kernel='rbf',probability=True,random_state=2022)
dtc= DecisionTreeClassifier(random_state=2022)

#----------------------------------------------------
logreg.fit(xtrain,ytrain)
y_pred = logreg.predict(xtrain)
acc_logreg = accuracy_score(ytrain, y_pred)
print(accuracy_score(ytrain, y_pred))
# 0.813692480359147
#----------------------------------------------------
#----------------------------------------------------
svm_l.fit(xtrain,ytrain)
y_pred = svm_l.predict(xtrain)
acc_svm_l = accuracy_score(ytrain, y_pred)
print(accuracy_score(ytrain, y_pred))
# 0.7968574635241302
#----------------------------------------------------
#----------------------------------------------------
svm_c.fit(xtrain,ytrain)
y_pred = svm_c.predict(xtrain)
acc_svm_c = accuracy_score(ytrain, y_pred)
print(accuracy_score(ytrain, y_pred))
#0.6823793490460157
#----------------------------------------------------
#----------------------------------------------------
lda.fit(xtrain,ytrain)
y_pred = lda.predict(xtrain)
acc_lda = accuracy_score(ytrain, y_pred)
print(accuracy_score(ytrain, y_pred))
# 0.8013468013468014
#----------------------------------------------------
#----------------------------------------------------
dtc.fit(xtrain,ytrain)
y_pred = dtc.predict(xtrain)
acc_dtc = accuracy_score(ytrain, y_pred)
print(accuracy_score(ytrain, y_pred))
# 0.9820426487093153
#----------------------------------------------------

models5=[('logreg',logreg),('svm_l',svm_l),('svm_c',svm_c),('lda',lda),('dtc',dtc)]

voting5 = VotingClassifier(models5,voting='soft',weights=np.array([acc_logreg,acc_svm_l,acc_svm_c,acc_lda,acc_dtc]))

start_time5 = time.time()

result5 = cross_val_score(voting5,xtrain,ytrain,cv=kfold,scoring='accuracy',verbose=10)

print(f'time5 = {time.time()-start_time5}')

print(result5)
# [0.82122905 0.81460674 0.79213483 0.76404494 0.8258427 ]
# [0.82681564 0.80898876 0.78089888 0.7752809  0.80337079]

print(result5.mean())
# 0.8035716527524952
# 0.7990709936601594

start_time5 = time.time()

voting5.fit(xtrain,ytrain)

print(f'time5 = {time.time()-start_time5}')

y_pred_prob5=voting5.predict_proba(xtest)[:,1]

y_pred_train=voting5.predict(xtrain)

print(accuracy_score(ytrain, y_pred_train))
# 0.8731762065095399
# 0.8978675645342312

y_pred5=voting5.predict(xtest)


##############submission VotingClassifier with cross_val_score

submit=pd.read_csv('gender_submission.csv',index_col=0)

submit['Survived']=y_pred5

submit.to_csv('titanic_submission_05a.csv')


##############################################################################
##############################################################################
####### DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

dtc6= DecisionTreeClassifier(random_state=2022)

dtc6.fit(xtrain,ytrain)

y_pred = dtc6.predict(xtrain)

acc_dtc = accuracy_score(ytrain, y_pred)

print(accuracy_score(ytrain, y_pred))
# 0.9820426487093153

y_pred_prob6=dtc6.predict_proba(xtest)[:,1]

y_pred6=dtc6.predict(xtest)

##############submission VotingClassifier with cross_val_score

submit=pd.read_csv('gender_submission.csv',index_col=0)

submit['Survived']=y_pred6

submit.to_csv('titanic_submission_06.csv')

