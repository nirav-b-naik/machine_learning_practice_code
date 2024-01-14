#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:28:18 2022

@author: dai
"""

import numpy as np
import pandas as pd
import surprise

df = pd.read_csv("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/Amazon Movie Ratings/Amazon.csv")

ratings = pd.melt(df,id_vars='user_id',var_name="item_id",value_name="rating")

ratings = ratings[ratings["rating"].notna()]