#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:19:06 2022

@author: dai
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df1=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/pizza.csv')

sns.scatterplot(df1,x='Promote',y='Sales')
plt.show()

df1['Promote'].corr(df1['Sales'])

df1.corr()

lr=LinearRegression()

x=df1['Promote']
y=df1['Sales']

x=x.values

x=x.reshape(-1,1)

lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)
