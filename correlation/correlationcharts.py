#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:19:06 2022

@author: dai
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/pizza.csv')

sns.scatterplot(df1,x='Promote',y='Sales')
plt.show()

df1['Promote'].corr(df1['Sales'])

df1.corr()

############################################################

insure = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Insure_auto.csv",index_col=0)
insure.corr()

sns.pairplot(insure) # Metrics plot
sns.heatmap(df1.corr())
sns.heatmap(insure.corr(),annot=True)

######################################

concrete=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Concrete Strength/Concrete_Data.csv')

sns.heatmap(concrete.corr(), annot=True)