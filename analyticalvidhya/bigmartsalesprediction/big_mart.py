import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Training\AV\Big Mart III")

train = pd.read_csv("train_v9rqX0R.csv")
print(train.columns)
print(train.info())

# Item_Identifier
train['Item_Identifier'].value_counts()

# Item_Weight
train['Item_Weight'].describe()

# Outlet Size
train['Outlet_Size'].value_counts()

# Item_Fat_Content
prev = train['Item_Fat_Content'].value_counts()
train['Item_Fat_Content'].replace({'reg':'Regular',
                                   'LF':'Low Fat',
                                   'low fat':'Low Fat'},
                                  inplace=True)
later = train['Item_Fat_Content'].value_counts()

# Item_Visibility
train['Item_Visibility'].describe()

# Item_Type
train['Item_Type'].value_counts()

# Imputing Item Weights
items = train[['Item_Identifier', 
               'Item_Weight']].sort_values(by='Item_Identifier')

weights_nonmissing = items[items['Item_Weight'].notna()]
weights_nonmissing_nodup = weights_nonmissing.drop_duplicates()
weights_nonmissing_nodup.rename({'Item_Identifier':'Item_Identifier',
                                 'Item_Weight':'i_weight'},
                                axis=1, inplace=True)
train_wt = weights_nonmissing_nodup.merge(train, how='outer',
                                          on='Item_Identifier')

train_wt.drop('Item_Weight', axis=1, inplace=True)

# Imputing Outlet Size
outlets = train[['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']]
outlets_nodup = outlets.drop_duplicates()

cnts = train.groupby(['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type'],
                     dropna=False)['Outlet_Type'].count()

sizes = outlets_nodup[['Outlet_Identifier','Outlet_Size']]
sizes.iloc[2,1] = "Small"
sizes.iloc[5,1] = "Small"
sizes.loc[9,"Outlet_Size"] = "Medium"
sizes.columns = ['Outlet_Identifier','O_Size']

train_wt_out = train_wt.merge(sizes, on="Outlet_Identifier")
train_wt_out.drop('Outlet_Size', axis=1, inplace=True)

#################### Test Set Finding ###########################
test = pd.read_csv("test_AbJTz2l.csv")
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
