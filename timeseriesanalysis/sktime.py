import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.utils.plotting import plot_series


os.chdir('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets')

mydata=pd.read_csv('BUNDESBANK-BBK01_WT5511.csv')

y=mydata['Value']

y_train,y_test = temporal_train_test_split(mydata,test_size=0.1)
fh = np.arange(1,len(y_test)+1) # forecasting horizon
regressor  =RandomForestRegressor(random_state=2022)
forecaster = make_reduction(regressor,window_length=10)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
rmse = MeanSquaredError(square_root=True)
print(rmse(y_test,y_pred))

plot_series(y_train,y_test,y_pred,labels=["Train","Test","Forecast"])
plt.show()
    
#plot results
plot_series(y_test,y_pred,labels=["Test","Forecast"])
plt.show()
