#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:35:39 2022

@author: dai
"""

import os
os.chdir("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets")

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("FRED-NROUST.csv")
df.head()

from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt

df.plot.line(x = 'Date',y = 'Value')
plt.show()

y = df['Value']
y_train = df['Value'][:-20]
y_test = df['Value'][-20:]

#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

span = 5
#### Trailing MA
fcast = y_train.rolling(span).mean()
MA = fcast.iloc[-1]
MA_series = pd.Series(MA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(MA_fcast, label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, MA_series))
print(rms)

alpha = 0.1
# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(y_train).fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test)).rename(r'$\alpha=0.1$')
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast1))
print(rms)


# Holt's Method
alpha = 0.9
beta = 0.02
### Linear Trend
fit1 = Holt(y_train).fit(smoothing_level=alpha, 
                         smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test)).rename("Holt's linear trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast1))
print(rms)

### Exponential Trend
alpha = 0.9
beta = 0.02
fit2 = Holt(y_train, exponential=True).fit(smoothing_level=alpha, 
                                           smoothing_trend=beta)
fcast2 = fit2.forecast(len(y_test)).rename("Exponential trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast2.plot(color="purple", label='Forecast')
plt.show()
rms = sqrt(mean_squared_error(y_test, fcast2))
print(rms)

### Additive Damped Trend
fit3 = Holt(y_train, damped_trend=True).fit()
fcast3 = fit3.forecast(len(y_test)).rename("Additive damped trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')

rms = sqrt(mean_squared_error(y_test, fcast3))
print(rms)

### Multiplicative Damped Trend
fit3 = Holt(y_train,exponential=True, damped_trend=True).fit()
fcast3 = fit3.forecast(len(y_test)).rename("Multiplicative damped trend")

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')

rms = sqrt(mean_squared_error(y_test, fcast3))
print(rms)


# Holt-Winters' Method

########### Additive #####################
fit1 = ExponentialSmoothing(y_train, seasonal_periods=4, 
                            trend='add', seasonal='add').fit()

fcast1 = fit1.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, fcast1))
print(rms)

########### Multiplicative #####################
fit2 = ExponentialSmoothing(y_train, seasonal_periods=4, trend='add', 
                            seasonal='mul').fit()

fcast2 = fit2.forecast(len(y_test)).rename("Holt-Winters Additive Trend and Multiplicative seasonality")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast2.plot(color="purple", label='Forecast')
plt.legend(loc='best')
rms = sqrt(mean_squared_error(y_test, fcast2))
print(rms)

########### Seasonal Additive & Damped #####################
fit3 = ExponentialSmoothing(y_train, seasonal_periods=4, trend='add', 
                            seasonal='add', damped_trend=True).fit()

fcast3 = fit3.forecast(len(y_test)).rename("Holt-Winters Additive Trend and seasonality with damping")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
rms = sqrt(mean_squared_error(y_test, fcast3))
print(rms)

########### Seasonal Multiplicative & Damped #####################
fit4 = ExponentialSmoothing(y_train, seasonal_periods=4, 
                            trend='add', seasonal='mul', 
                            damped_trend=True).fit()

fcast4 = fit4.forecast(len(y_test)).rename("Holt-Winters Multiplicative Trend and Multiplicative seasonality with damping")
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast4.plot(color="purple", label='Forecast')

rms = sqrt(mean_squared_error(y_test, fcast4))
print(rms)
