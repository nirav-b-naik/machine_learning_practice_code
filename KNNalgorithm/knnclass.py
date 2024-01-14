import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.chdir(r"/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets")

df = pd.read_csv("RidingMowers.csv")
dum_df = pd.get_dummies(df, drop_first=True)

sns.scatterplot(x='Income', y='Lot_Size',
                hue='Response',
                data=df)
plt.show()

X = dum_df.drop("Response_Not Bought", axis=1)
y = dum_df["Response_Not Bought"]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=2022,
                                                    test_size=0.3,
                                                    stratify=y)

y.value_counts(normalize=True)*100
y_train.value_counts(normalize=True)*100
y_test.value_counts(normalize=True)*100

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ROC AUC
y_pred_prob = knn.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test, y_pred_prob))

y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Best k = 3
tst_mowers = pd.read_csv("testMowers.csv")
predictions = knn.predict(tst_mowers)

tst_mowers["predictions"]= predictions