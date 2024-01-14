#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:49:13 2022

@author: dai
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets')

mydata=pd.read_csv('BUNDESBANK-BBK01_WT5511.csv')

y=mydata['Value']

y_train=mydata['Value'][:-12]
y_test=mydata['Value'][-12:]

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(mydata['Value'],lags=20)
plt.show()

## ARIMA
from pmdarima.arima import auto_arima
model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True)
# Best model:  ARIMA(0,1,1)(0,0,0)[0] intercept

forecast = model.predict(n_periods=len(y_test))
import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.utils.plotting import plot_series


forecast = pd.DataFrame(forecast,index=y_test.index,columns=["Predictions"])

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test MSE: ',rms)
# 91.10508105558864

## SARIMA
model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   seasonal=True,m=12)
# Best model:  ARIMA(2,1,2)(0,0,2)[12] intercept

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index=y_test.index,columns=["Predictions"])

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test MSE: ',rms)
# 81.57187933276406

# Plot the prediction for validation set
plt.plot(y_train,label="train",color="blue")
plt.plot(y_test,label="test",color="pink")
plt.plot(forecast,label="Prediction",color="purple")
plt.show()

#plot results
plt.plot(y_test)
plt.plot(forecast,color="red")
plt.show()
