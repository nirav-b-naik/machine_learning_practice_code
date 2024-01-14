#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:04:05 2022

@author: dai
"""

import numpy as np
import pandas as pd
import os

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets")
exp_salary = pd.read_csv("Exp_Salaries.csv")
print(exp_salary.info())

dum_exp = pd.get_dummies(exp_salary,drop_first=True)
dum_exp1 = pd.get_dummies(exp_salary['Gender'],drop_first=True)

#############################################################
jobsal=pd.read_csv("JobSalary2.csv")

#droping rows with NaN values
jobsal.dropna()

#constant imputation
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='constant',fill_value=50)

imp_data=imputer.fit_transform(jobsal)

imp_pd_data=pd.DataFrame(imp_data,columns=jobsal.columns)

#mean imputation
imputer=SimpleImputer(strategy='mean')

imp_data=imputer.fit_transform(jobsal)

imp_pd_data=pd.DataFrame(imp_data,columns=jobsal.columns)

#median imputation
imputer=SimpleImputer(strategy='median')

imp_data=imputer.fit_transform(jobsal)

imp_pd_data=pd.DataFrame(imp_data,columns=jobsal.columns)
