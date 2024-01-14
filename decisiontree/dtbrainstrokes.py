#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:37:21 2022

@author: dai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:44:23 2022

@author: dai
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/kaggledatasets/Brainstrokeprediction/full_data.csv")
dummy_df=pd.get_dummies(df,drop_first=True)
x = dummy_df.drop('stroke',axis=1)
y = dummy_df["stroke"]

clf =  DecisionTreeClassifier(random_state=2022)

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {"max_depth":[None,3,5,10],
          "min_samples_split":[2,5,10,50],
          "min_samples_leaf":[1,5,10,50]}

gcv = GridSearchCV(clf, param_grid=params,cv=kfold,scoring='roc_auc')

gcv.fit(x,y)

print(gcv.best_params_)
print(gcv.best_score_)

gcv_model=gcv.best_estimator_

plt.figure(figsize=(40,30))
plot_tree(gcv_model,feature_names=x.columns,
          class_names=['1', '0'],
          filled=True,fontsize=10)

plt.figure()
print(gcv_model.feature_importances_)
ind = np.arange(14)
plt.bar(ind,gcv_model.feature_importances_)
plt.xticks(ind,x.columns,rotation=90)
plt.title('feature importance')
plt.xlabel('variables')
plt.show()