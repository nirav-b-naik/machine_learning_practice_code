#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:59:54 2022

@author: dai
"""
########################################################################
###################PENDINGGGGGGGGGGGGGGGGG
###########################################################################
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets')

mydata=pd.read_csv('FMAC-HPI_24420.csv')

y=mydata['NSA Value']

y_train=mydata['NSA Value'][:-6]
y_test=mydata['NSA Value'][-6:]

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(mydata['NSA Value'],lags=20)
plt.show()

from pmdarima.arima import auto_arima
model = auto_arima(y,trace=True,error_action='ignore',suppress_warnings=True)

forecast = model.predict(n_periods = len(y))
#forecast = pd.DataFrame(forecast,index=y.index,columns=["Predictions"])
rms = sqrt(mean_squared_error(y, forecast))
print('Test MSE: ',rms)
