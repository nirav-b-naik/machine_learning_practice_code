#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:07:37 2022

@author: dai
"""
import numpy as np
import pandas as pd
import h2o

h2o.init()

bank = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/bank/bank.csv",sep=";")
dum_bnk = pd.get_dummies(bank,drop_first=True)

h2o_bnk = h2o.H2OFrame(dum_bnk)
print(h2o_bnk.col_names)

all_columns = h2o_bnk.col_names

x = all_columns[:-1]
y = "y_yes"

h2o_bnk["y_yes"] = h2o_bnk["y_yes"].asfactor()
print(h2o_bnk["y_yes"].levels())

train,test = h2o_bnk.split_frame(ratios=[0.7],seed=2022)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

glm_logistic = H2OGeneralizedLinearEstimator(family="binomial")

glm_logistic.train(x=x,y=y,training_frame=train,validation_frame=test,model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix())
print(glm_logistic.auc()) # 0.9020416938701844

#h2o.cluster().shutdown()



##############################################################################

bank = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/bank/bank-full.csv",sep=";")
dum_bnk = pd.get_dummies(bank,drop_first=True)

h2o_bnk = h2o.H2OFrame(dum_bnk)
print(h2o_bnk.col_names)

all_columns = h2o_bnk.col_names

x = all_columns[:-1]
y = "y_yes"

h2o_bnk["y_yes"] = h2o_bnk["y_yes"].asfactor()
print(h2o_bnk["y_yes"].levels())

train,test = h2o_bnk.split_frame(ratios=[0.7],seed=2022)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

glm_logistic = H2OGeneralizedLinearEstimator(family="binomial")

glm_logistic.train(x=x,y=y,training_frame=train,validation_frame=test,model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix())
print(glm_logistic.auc()) # 0.9078232866544504