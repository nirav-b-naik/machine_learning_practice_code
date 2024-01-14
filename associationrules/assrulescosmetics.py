#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:26:16 2022

@author: dai
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df=pd.read_csv('/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Cosmetics.csv',index_col=0)

itemfrequency=df.sum(axis=0)

itemset=apriori(df,min_support=0.3,use_colnames=True)

rules=association_rules(itemset,min_threshold=0.6,metric='confidence')
