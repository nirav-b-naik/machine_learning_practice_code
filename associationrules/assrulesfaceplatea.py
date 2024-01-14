#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:04:40 2022

@author: dai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

fp_df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Datasets/Faceplate.csv",index_col=0)

#Support of 1-tem freq set
itemFrequency = fp_df.sum(axis=0)
plt.bar(itemFrequency.index,itemFrequency)
plt.show()

# create frequent item sets
itemsets = apriori(fp_df,min_support=0.2,use_colnames=True)

# and convert into rules
rules = association_rules(itemsets,metric="confidence",min_threshold=0.4)
