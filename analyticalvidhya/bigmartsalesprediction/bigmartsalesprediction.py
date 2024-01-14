#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:47:26 2022

@author: dai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train= pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/analyticalvidhya/bigmartsalesprediction/train_v9rqX0R.csv")

test= pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/analyticalvidhya/bigmartsalesprediction/test_AbJTz2l.csv")

train.isna().sum()

train['Item_Fat_Content'].replace({'reg':'Regular',
                                'LF':'Low Fat',
                                'low fat':'Low Fat'},inplace=True)

###################################### Item_Weight 

train['Item_Weight'].corr(train['Item_Visibility'])

items=train[['Item_Identifier','Item_Weight']]

weights_missing=items[items['Item_Weight'].notna()]

weights_missing_nodup=weights_missing.drop_duplicates()

weights_missing_nodup.rename({'Item_Identifier':'Item_Identifier',
                              'Item_Weight':'i_weight'},
                             axis=1, inplace=True)

train_wt=weights_missing_nodup.merge(train,how='outer',
                                     on='Item_Identifier')

train_wt.drop('Item_Weight',axis=1,inplace=True)


###################################### Outlet_Size 

outlets=train[['Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 
               'Outlet_Location_Type','Outlet_Type']]

outlets_nodup=outlets.drop_duplicates()

cnts=train.groupby(['Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 
               'Outlet_Location_Type','Outlet_Type'],dropna=False)['Outlet_Type'].count()

sizes=outlets_nodup[['Outlet_Identifier','Outlet_Size']]

sizes.iloc[0,1]='Small'
sizes.iloc[2,1]='Small'
sizes.iloc[7,1]='Medium'

sizes.columns=['Outlet_Identifier','O_Size']

train_wt_out=train_wt.merge(sizes,on='Outlet_Identifier')

train_wt_out.drop('Outlet_Size',axis=1,inplace=True)

#################### Test Set Finding ###########################

# missings in train
train_wt[train_wt['i_weight'].isna()]['Item_Identifier'].unique()
p_test = test[test['Item_Identifier'].isin(['FDN52', 'FDK57', 'FDE52', 'FDQ60'])]
p_test = p_test[['Item_Identifier','Item_Weight']].drop_duplicates().dropna()

####### Back Merging on train set

train_wt_out = train_wt_out.join(p_test.set_index('Item_Identifier'),
                                 on='Item_Identifier')
train_wt_out['i_weight'] = np.where(train_wt_out['i_weight'].isna(),
         train_wt_out['Item_Weight'],
         train_wt_out['i_weight'])

train_wt_out.drop('Item_Weight', axis=1, inplace=True)


#################### Find hterelation between the following variables

##### 1. item weight, item sales

train_wt_out['i_weight'].corr(train_wt_out['Item_Outlet_Sales'])

# pearsons coefficient method

from scipy.stats import pearsonr

pearsonr(train_wt_out['i_weight'], train_wt_out['Item_Outlet_Sales'])


##### 2. item type, item sales

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# anova test
"""
Null Hypothesis: H0: m1=m2=m3=m4=.......=m16 mean of Item_Outlet_Sales of all Item_Type are same.
Alternate Hypothesis: Ha: Mean of Item_Outlet_Sales of all Item_Type are not same.
"""

aov=ols('Item_Outlet_Sales ~ Item_Type', data=train_wt_out).fit()

table=anova_lm(aov, type=2)

print(table)

# from the above table pvalue is <0.05 hance reject null hypothesis.

# bar plot for Item_Outlet_Sales ~ Item_Type
train_wt_out.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values().plot(kind='barh')


##### 3. Item_Type, O_Size

from scipy.stats import chi2_contingency
import seaborn as sns

"""
Null Hypothesis: H0: O_Size and Item_Type are '''independent'''.
Alternate Hypothesis: Ha: O_Size and Item_Type are '''dependent'''.
"""

table=pd.crosstab(train_wt_out['Item_Type'], train_wt_out['O_Size'])

chi2,pvalue,dof,expected=chi2_contingency(table)

# from the above table pvalue is >0.05 hance accept null hypothesis.

# bar plot for O_Size ~ Item_Type
train_wt_out.groupby('Item_Type')['O_Size'].nunique().value_counts().plot()

table.plot(kind='barh')

table=table.reset_index()

molten=pd.melt(table,id_vars='Item_Type',value_name='Count')

sns.catplot(molten,
            y='Item_Type',
            x='Count',
            kind='bar',hue='O_Size')

##### 4. Outlet_Type, Item_Type

"""
Null Hypothesis: H0: O_Size and Item_Type are '''independent'''.
Alternate Hypothesis: Ha: O_Size and Item_Type are '''dependent'''.
"""

table=pd.crosstab(train_wt_out['Item_Type'], train_wt_out['Outlet_Type'])

chi2,pvalue,dof,expected=chi2_contingency(table)

# from the above table pvalue is >0.05 hance accept null hypothesis.

# bar plot for O_Size ~ Item_Type
train_wt_out.groupby('Item_Type')['Outlet_Type'].nunique().value_counts().plot()

table.plot(kind='barh')

table=table.reset_index()

molten=pd.melt(table,id_vars='Item_Type',value_name='Count')

sns.catplot(molten,
            y='Item_Type',
            x='Count',
            kind='bar',hue='Outlet_Type')


##### 5. Outlet_Type, item sales

# anova test
"""
Null Hypothesis: H0: m1=m2=m3=m4=.......=m16 mean of Item_Outlet_Sales of all Outlet_Type are same.
Alternate Hypothesis: Ha: Mean of Item_Outlet_Sales of all Outlet_Type are not same.
"""

aov=ols('Item_Outlet_Sales ~ Outlet_Type', data=train_wt_out).fit()

table=anova_lm(aov, type=2)

print(table)

# from the above table pvalue is <0.05 hance reject null hypothesis.

# bar plot for Item_Outlet_Sales ~ Item_Type
train_wt_out.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values().plot(kind='barh')


##### 6. fat content, item sales

# anova test
"""
Null Hypothesis: H0: m1=m2=m3=m4=.......=m16 mean of Item_Outlet_Sales of all Item_Fat_Content are same.
Alternate Hypothesis: Ha: Mean of Item_Outlet_Sales of all Item_Fat_Content are not same.
"""

aov=ols('Item_Outlet_Sales ~ Item_Fat_Content', data=train_wt_out).fit()

table=anova_lm(aov, type=2)

print(table)

# from the above table pvalue is >0.05 hance accept null hypothesis.

# bar plot for Item_Outlet_Sales ~ Item_Type
train_wt_out.groupby('Item_Fat_Content')['Item_Outlet_Sales'].mean().sort_values().plot(kind='barh')
 

##### 7. Item_MRP, Item_Outlet_Sales

train_wt_out['Item_MRP'].corr(train_wt_out['Item_Outlet_Sales'])

# pearsons coefficient method

pearsonr(train_wt_out['Item_MRP'], train_wt_out['Item_Outlet_Sales'])

sns.scatterplot(data=train_wt_out,x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content')


###############################################################################
########################## test dataframe

train_item=train_wt_out['Item_Identifier'].unique()
test_item=test['Item_Identifier'].unique()

np.setdiff1d(train_item, test_item)


test['Item_Fat_Content'].replace({'reg':'Regular',
                                'LF':'Low Fat',
                                'low fat':'Low Fat'},inplace=True)

###################################### Item_Weight 

test['Item_Weight'].corr(test['Item_Visibility'])

items=test[['Item_Identifier','Item_Weight']]

weights_missing=items[items['Item_Weight'].notna()]

weights_missing_nodup=weights_missing.drop_duplicates()

weights_missing_nodup.rename({'Item_Identifier':'Item_Identifier',
                              'Item_Weight':'i_weight'},
                             axis=1, inplace=True)

test_wt=weights_missing_nodup.merge(test,how='outer',
                                     on='Item_Identifier')

test_wt.drop('Item_Weight',axis=1,inplace=True)

t_itm_id=test_wt[test_wt['i_weight'].isna()]['Item_Identifier'].drop_duplicates()


###################################### Outlet_Size 

outlets=test[['Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 
               'Outlet_Location_Type','Outlet_Type']]

outlets_nodup=outlets.drop_duplicates()

cnts=test.groupby(['Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 
               'Outlet_Location_Type','Outlet_Type'],dropna=False)['Outlet_Type'].count()

sizes=outlets_nodup[['Outlet_Identifier','Outlet_Size']]

sizes.iloc[1,1]='Medium'
sizes.iloc[2,1]='Small'
sizes.iloc[6,1]='Small'

sizes.columns=['Outlet_Identifier','O_Size']

test_wt_out=test_wt.merge(sizes,on='Outlet_Identifier')

test_wt_out.drop('Outlet_Size',axis=1,inplace=True)

##### missings in test
test_wt[test_wt['i_weight'].isna()]['Item_Identifier'].unique()
p_train = train_wt[train_wt['Item_Identifier'].isin(['FDL58', 'FDY57', 'FDH58', 'FDI45', 'FDG50', 'FDG57', 'FDJ09',
                                             'FDF22', 'DRN47', 'NCJ30', 'FDT21', 'FDO22', 'FDG09', 'FDF05',
                                             'FDP28', 'FDF04'])]

p_train = p_train[['Item_Identifier','i_weight']].drop_duplicates().dropna()
p_train.columns=['Item_Identifier','Item_Weight']

###### Back Merging on train set

test_wt_out = test_wt_out.join(p_train.set_index('Item_Identifier'),
                                 on='Item_Identifier')
test_wt_out['i_weight'] = np.where(test_wt_out['i_weight'].isna(),
         test_wt_out['Item_Weight'],
         test_wt_out['i_weight'])

test_wt_out.drop('Item_Weight', axis=1, inplace=True)


###############################################################################
########################## Building a ML alogrithm
###############################################################################

from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score

x=train_wt_out.drop(['Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales'],axis=1)

xtest=test_wt_out.drop(['Item_Identifier', 'Outlet_Identifier'],axis=1)

x=pd.get_dummies(x)

xtest=pd.get_dummies(xtest)

y=train_wt_out['Item_Outlet_Sales']

gbm=XGBRegressor(random_state=2022)

kfold=KFold(n_splits=5,shuffle=True, random_state=2022)

params = {"learning_rate":np.linspace(0.001,0.5,10),
          "max_depth":[None,2,3,4,5,6],
          "n_estimators":[50,100,150]}

gcv=GridSearchCV(gbm, param_grid=params, cv=kfold, verbose=10,scoring='r2')

gcv.fit(x,y)

print(gcv.best_params_)
# {'learning_rate': 0.112, 'max_depth': 3, 'n_estimators': 50}
# {'learning_rate': 0.05644444444444444, 'max_depth': 2, 'n_estimators': 150}

print(gcv.best_score_)
# 0.5979162959450457
# 0.5984681953575778

best_model=gcv.best_estimator_

imp=best_model.feature_importances_

i_sort=np.argsort(-imp)
sorted_imp=imp[i_sort]
sorted_col=x.columns[i_sort]

ind=np.arange(x.shape[1])
plt.barh(ind,sorted_imp)

plt.show()

y_train_pred=best_model.predict(x)

print(r2_score(y, y_train_pred))
# 0.6123984699506022


y_test_pred=best_model.predict(xtest)

y_test_pred[y_test_pred<0]=0

##############submission XGBRegressor with gridsearchcv

submit=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/analyticalvidhya/bigmartsalesprediction/sample_submission_8RXa3c6.csv',index_col=0)

submit['Item_Outlet_Sales']=y_test_pred

submit.to_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/analyticalvidhya/bigmartsalesprediction/submission_xgb_01.csv')















