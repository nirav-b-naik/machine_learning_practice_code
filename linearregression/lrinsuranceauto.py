#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 08:55:00 2022

@author: dai
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


insure = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Insure_auto.csv",index_col=0)
insure.corr()

sns.pairplot(insure) # Metrics plot
sns.heatmap(insure.corr(),annot=True)

insure.columns

lr=LinearRegression()

x=insure.drop('Operating_Cost',axis=1)
y=insure['Operating_Cost']

lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)

