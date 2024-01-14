import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Vehicle Silhouettes")
veh_seg = pd.read_csv("Vehicle.csv")

x=veh_seg.drop('Class',axis=1)
y=veh_seg['Class']

le=LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

x_train, x_test, y_train, y_test=train_test_split(x,
                                                  le_y,
                                                  test_size=0.3,
                                                  random_state=2022,
                                                  stratify=y)

# for standard scaler
scaler=StandardScaler()

# for minmax scaler
scaler=MinMaxScaler()

i=0
logloss=list()

for i in range(1,31):
    knn=KNeighborsClassifier(n_neighbors=i)
    pipe= Pipeline([('scl_mm',scaler),('knn_model',knn)])
    pipe.fit(x_train,y_train)
    
    y_pred_prob=pipe.predict_proba(x_test)
    
    y_pred=pipe.predict(x_test)  
   # print(f"{i}: k: Log_Loss_score:",round(log_loss(y_test, y_pred_prob),6))
    logloss.append(round(log_loss(y_test, y_pred_prob),6))
    
print(f'best k= {np.argmin(logloss)+1} with logloss= {np.min(logloss)}')
print(f'k= {np.argmax(logloss)+1} with max logloss= {np.max(logloss)}')

