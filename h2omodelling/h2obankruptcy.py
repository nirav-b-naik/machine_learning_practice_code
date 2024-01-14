#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 08:56:02 2022

@author: dai
"""

import h2o

#start h2o engine
h2o.init()

#loding data into h2o dataframe

df=h2o.import_file('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Bankruptcy/Bankruptcy.csv',
                   destination_frame='Bankruptcy')

print(df.col_names)

#adding response in y in form of list.
y='D'

#adding features in x in form of list.
x= df.col_names[3:]

#to consider as categorical response.
df['D']=df['D'].asfactor()

#show unique values of response.
print(df['D'].levels())

#train test split on ratios
train,test=df.split_frame(ratios=[0.7],seed=2022)

print(df.shape)

print(train.shape)

print(test.shape)

# calling estimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# creating object of estimator or model
glm_logistic=H2OGeneralizedLinearEstimator(family='binomial')

# training the model on train and validating it with test
glm_logistic.train(x=x,y=y,training_frame=train,validation_frame=test,model_id='glm_logistic')

# predicting the values on test data
y_pred = glm_logistic.predict(test_data=test)

# converting to pandas dataframe
y_pred_df = y_pred.as_data_frame()

# generating confusion matrix
print(glm_logistic.confusion_matrix())

# finding AUC
print(glm_logistic.auc())


# print data of model perfonace
print(glm_logistic.model_performance())

# terminating the engine
h2o.cluster().shutdown()


