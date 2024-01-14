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
from sklearn.preprocessing import PolynomialFeatures

##########################pizza dataset

df1=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/pizza.csv')

sns.scatterplot(df1,x='Promote',y='Sales')
plt.show()

df1['Promote'].corr(df1['Sales'])

df1.corr()

x = df1['Promote']
y = df1["Sales"]
lr = LinearRegression()

x=x.values

x=x.reshape(-1,1)

poly  =PolynomialFeatures(degree=2)
poly_x =poly.fit_transform(x)
print(poly.get_feature_names_out())

pd_poly_x = pd.DataFrame(poly_x,columns=poly.get_feature_names_out())

lr.fit(pd_poly_x,y)

print(lr.intercept_)
print(lr.coef_)

##########################insurance auto

insure = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Insure_auto.csv",index_col=0)
insure.corr()

x=insure.drop('Operating_Cost',axis=1)
y=insure['Operating_Cost']

lr = LinearRegression()

poly = PolynomialFeatures(degree=2)
poly_x = poly.fit_transform(x)
print(poly.get_feature_names_out())

pd_poly_x = pd.DataFrame(poly_x,columns = poly.get_feature_names_out())

lr.fit(pd_poly_x,y)

print(lr.intercept_)
print(lr.coef_)
