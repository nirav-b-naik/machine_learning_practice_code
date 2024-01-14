#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:41:09 2022

@author: dai
"""

import numpy as np
import pandas as pd
import h2o

h2o.init()

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

############################ with H2ORandomForestEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator

glm_logistic = H2ORandomForestEstimator(seed=2022)

glm_logistic.train(x=x,y=y,training_frame=train,validation_frame=test,model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix())
print(glm_logistic.auc()) # 0.9189791834673008


############################ with H2OXGBoostEstimator

from h2o.estimators.xgboost import H2OXGBoostEstimator

glm_logistic = H2OXGBoostEstimator(seed=2022)

glm_logistic.train(x=x,y=y,training_frame=train,validation_frame=test,model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix())
print(glm_logistic.auc()) # 0.9675497747940845


############################ with H2OGradientBoostingEstimator

from h2o.estimators.gbm import H2OGradientBoostingEstimator

glm_logistic = H2OGradientBoostingEstimator(seed=2022)

glm_logistic.train(x=x,y=y,training_frame=train,validation_frame=test,model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.confusion_matrix())
print(glm_logistic.auc()) # 0.9385858110885252
