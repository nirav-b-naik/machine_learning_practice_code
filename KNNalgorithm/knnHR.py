import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    
    from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
    from sklearn.neighbors import KNeighborsClassifier
    
    for i in range(1,11,2):
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train,y_train)
        
        #ROC AUC
        y_pred_prob=knn.predict_proba(x_test)[:,1]
        
        
        y_pred=knn.predict(x_test)  
        print(f"{lst}:{i}:roc_auc_score: ",round(roc_auc_score(y_test, y_pred_prob),5),end="\t\t")
        print(f"{i}:accuracy_score: ",round(accuracy_score(y_test, y_pred),5))
