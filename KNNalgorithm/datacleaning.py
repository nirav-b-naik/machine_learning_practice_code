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

#total NA
jobsal.isnull().sum()
jobsal.isnull().sum().sum()

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

#meidan imputation
imputer=SimpleImputer(strategy='median')
imp_data=imputer.fit_transform(jobsal)
imp_pd_data=pd.DataFrame(imp_data,columns=jobsal.columns)

############################################################

chemprocess=pd.read_csv("ChemicalProcess.csv")

chemprocess.isnull().sum()
chemprocess.isnull().sum().sum()

#median imputation
imputer=SimpleImputer(strategy='median')
imp_data=imputer.fit_transform(chemprocess)
imp_pd_data=pd.DataFrame(imp_data,columns=chemprocess.columns)

imp_pd_data.isnull().sum().sum()

############################################################
from sklearn.preprocessing import StandardScaler,MinMaxScaler
milk = pd.read_csv("milk.csv",index_col=0)
scl_std =StandardScaler()
scl_std.fit(milk)

#Means
print(scl_std.mean_)

#Std Dev
print(scl_std.scale_)

trns_scl = scl_std.transform(milk)
# or
trns_scl = scl_std.fit_transform(milk)
type(trns_scl)
trans_scl_pd = pd.DataFrame(trns_scl,columns=milk.columns,index=milk.index)
trans_scl_pd.mean()
trans_scl_pd.std()

############################################################
milk=pd.read_csv('milk.csv',index_col=0)
scl_mm = MinMaxScaler()
scl_mm.fit(milk)

#Min
print(scl_mm.data_min_)
#Max
print(scl_mm.data_max_)

trns_scl = scl_mm.transform(milk)
#or
trns_scl = scl_mm.fit_transform(milk)
type(trns_scl)

trns_scl_pd = pd.DataFrame(trns_scl,columns=milk.columns,index=milk.index)
trns_scl_pd.min()
trns_scl_pd.max()





