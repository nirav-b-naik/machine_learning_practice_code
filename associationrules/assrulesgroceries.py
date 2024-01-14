#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:52:00 2022

@author: dai
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries=[]

with open('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Groceries.csv','r') as f: groceries=f.read()

groceries=groceries.split('\n')

groceries_list=[]

for i in groceries:
    groceries_list.append(i.split(','))

print(groceries_list)

te=TransactionEncoder()

te_ary=te.fit(groceries_list).transform(groceries_list)

df=pd.DataFrame(te_ary,columns=te.columns_)



itemfrequency=df.sum(axis=0)

itemset=apriori(df,min_support=0.05,use_colnames=True)

rules=association_rules(itemset,min_threshold=0.05,metric='confidence')
print(rules)
