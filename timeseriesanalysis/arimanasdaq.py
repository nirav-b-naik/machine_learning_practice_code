#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:25:29 2022

@author: dai
"""

import pandas as pd
import matplotlib.pyplot as plt
import nasdaqdatalink

mydata=nasdaqdatalink.get("LBMA/GOLD", authtoken="srM_ctoZEpefQ4Aaf9vd")

mydata.plot.line(y='USD (AM)')

mydata.reset_index(inplace=True)

y=mydata['USD (AM)']

y=y.ffill(inplace=True)

y_train=mydata['USD (AM)'][:-10]
y_test=mydata['USD (AM)'][-10:]

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(mydata['USD (AM)'],lags=20)
plt.show()

## ARIMA
from pmdarima.arima import auto_arima
model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True)
# ARIMA(3,1,2)(0,0,0)[0] intercept

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index=y_test.index,columns=["Predictions"])

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test MSE: ',rms)
# 83.12700861058707


## SARIMA
model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   seasonal=True,m=5)
#Best model:  ARIMA(2,1,1)(2,0,2)[5] intercept

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast,index=y_test.index,columns=["Predictions"])

rms = sqrt(mean_squared_error(y_test, forecast))
print('Test MSE: ',rms)
# 83.0257401172778

# Plot the prediction for validation set
plt.plot(y_train,label="train",color="blue")
plt.plot(y_test,label="test",color="pink")
plt.plot(forecast,label="Prediction",color="purple")
plt.show()

#plot results
plt.plot(y_test)
plt.plot(forecast,color="red")
plt.show()
