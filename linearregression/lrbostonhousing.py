
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Boston Housing/Boston.csv")
x = df.drop('medv',axis=1)
y = df['medv']
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
lr = LinearRegression()
lr.fit(x, y)

result = cross_val_score(lr,x,y,cv=kfold,scoring='r2')
print(lr.coef_)
print(lr.intercept_)
print(result)
print("result",result.mean())
#################################################

knn = KNeighborsRegressor()
scaler = StandardScaler()
pipe = Pipeline([('scl_std',scaler),('knn_model',knn)])
param = {'knn_model__n_neighbors':np.arange(1,31)}
gcv = GridSearchCV(pipe, param_grid=param,scoring='r2',cv=kfold)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)