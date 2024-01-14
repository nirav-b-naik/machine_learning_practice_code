#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:02:10 2022

@author: dai
"""

import numpy as np
import pandas as pd
import surprise

df = pd.read_excel("/home/dai/Desktop/dai2022/modulewise/practicalmachinelearning/Cases/jester_dataset/[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx",header=None)
rating = pd.melt(df,id_vars=0) 
