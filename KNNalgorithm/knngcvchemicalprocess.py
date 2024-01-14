import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import os

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Chemical Process Data")
df=pd.read_csv('ChemicalProcess.csv')

x = df.drop('Yield',axis=1)
y = df['Yield']

imputer = SimpleImputer()
#StandardScaler()
scaler = StandardScaler()

#MinMaxScaler
scalerm = MinMaxScaler()

kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
knn = KNeighborsRegressor()

pipe = Pipeline([('knn_model',knn),('imputer',imputer),('scl_mm',scalerm)])

print(pipe.get_params())

#################
params = {'knn_model__n_neighbors':np.arange(1,31) , 'imputer__strategy':['mean','median']}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold,scoring='r2',verbose=1)
gcv.fit(x,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_gcv = pd.DataFrame(gcv.cv_results_)

