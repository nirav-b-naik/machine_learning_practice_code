import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/human-resources-analytics")
df = pd.read_csv("HR_comma_sep.csv")
dum_df=pd.get_dummies(df,drop_first=True)
ylist=['left']
#['Department_RandD', 'Department_accounting',
#'Department_hr', 'Department_management', 'Department_marketing',
#'Department_product_mng', 'Department_sales', 'Department_support',
#'Department_technical', 'salary_low', 'salary_medium']
for lst in ylist:
    x=dum_df.drop(ylist,axis=1)
    y=dum_df[lst]
    
    x_train, x_test, y_train, y_test=train_test_split(x,
                                                      y,
                                                      test_size=0.3,
                                                      random_state=2022,
                                                      stratify=y)
    
    # for standard scaler
    scaler=StandardScaler()
    scl_trn=scaler.fit_transform(x_train)
    scl_test=scaler.transform(x_test)
    
    for i in range(1,15,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(scl_trn,y_train)
        
        #ROC AUC
        y_pred_prob=knn.predict_proba(scl_test)[:,1]
        
        y_pred=knn.predict(scl_test)  
        print(f"{lst}:{i}:roc_auc_score:",round(roc_auc_score(y_test, y_pred_prob),6),end="\t\t")
        print("accuracy_score:",round(accuracy_score(y_test, y_pred),6))
    
    # for minmax scaler
    scaler=MinMaxScaler()
    scl_trn=scaler.fit_transform(x_train)
    scl_test=scaler.transform(x_test)
    
    for i in range(1,15,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(scl_trn,y_train)
        
        #ROC AUC
        y_pred_prob=knn.predict_proba(scl_test)[:,1]
        
        y_pred=knn.predict(scl_test)  
        print(f"{lst}:{i}:roc_auc_score:",round(roc_auc_score(y_test, y_pred_prob),6),end="\t\t")
        print("accuracy_score:",round(accuracy_score(y_test, y_pred),6))


############### Pipeline ################

    from sklearn.pipeline import Pipeline
    
    # for minmax scaler
    scaler=MinMaxScaler()
    for i in range(1,1000,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        
        pipe= Pipeline([('scl_mm',scaler),('knn_model',knn)])
        pipe.fit(x_train,y_train)
        
        #ROC AUC
        y_pred_prob=pipe.predict_proba(x_test)[:,1]
        
        y_pred=pipe.predict(x_test)  
        print(f"{lst}:{i}:roc_auc_score:",round(roc_auc_score(y_test, y_pred_prob),6),end="\t\t")
        print("accuracy_score:",round(accuracy_score(y_test, y_pred),6))