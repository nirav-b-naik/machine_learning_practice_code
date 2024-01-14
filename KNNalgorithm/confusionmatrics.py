import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_test=np.array([0,0,0,1,0,1,1,0,1,0,0,1])
y_pred=np.array([0,0,0,0,1,1,0,0,1,1,0,0])

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

##############################################################################

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

y_test=np.array([1,1,0,0,0,1,0,0,1,0])
y_prob1=np.array([0.6,0.4,0.8,0.2,0.3,0.6,0.3,0.8,0.6,0.3])
y_prob2=np.array([0.7,0.45,0.9,0.2,0.15,0.7,0.3,0.6,0.7,0.2])

one_m_spac, sensitivity, thresold = roc_curve(y_test, y_prob1)

auc=roc_auc_score(y_test, y_prob1)
plt.plot(one_m_spac,sensitivity)
plt.text(0.6,0.4,auc)
plt.show()

one_m_spac, sensitivity, thresold = roc_curve(y_test, y_prob2)
print(one_m_spac, sensitivity, thresold)
auc=roc_auc_score(y_test, y_prob2)
plt.plot(one_m_spac,sensitivity)
plt.text(0.6,0.4,auc)
plt.show()

print(log_loss(y_test, y_prob1))

print(log_loss(y_test, y_prob2))


y_test1=np.array([1,1,1,1,0,1,0,1,1,0])
y_prob3=np.array([0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99])

one_m_spac, sensitivity, thresold = roc_curve(y_test1, y_prob3)

auc=roc_auc_score(y_test1, y_prob3)
plt.plot(one_m_spac,sensitivity)
plt.text(0.6,0.4,auc)
plt.show()

print(log_loss(y_test1, y_prob3))
