#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:28:45 2022

@author: dai
"""

import pandas as pd
import numpy as np
import surprise

rating=pd.read_csv('/home/dai/Desktop/dai2022/STUDENT FOLDER/STUDY MATERIAL AND ASSIGMENTS/5. PML /1. Day wise Study Material /20. Recommender Systems/filmtrust/ratings.txt',sep=' ',names=['uid','iid','rating'])

rating.head()

lowest_rating = rating['rating'].min()

highest_rating = rating['rating'].max()

print("Ratings range between {0} and {1}".format(lowest_rating,highest_rating))

reader = surprise.Reader(rating_scale = (lowest_rating,highest_rating))

data = surprise.Dataset.load_from_df(rating,reader)

similarity_options={'name':'cosine','user_based':True}

algo=surprise.KNNBasic(sim_options=similarity_options)

output=algo.fit(data.build_full_trainset())

pred=algo.predict(uid=60, iid=217)
score=pred.est

print(score)
    
iids=rating['iid'].unique()

rec_50=rating[rating['uid']==1250]

iids50=rec_50['iid']

iids_to_predict=np.setdiff1d(iids, iids50)

testset=[[1250,iid,0.0] for iid in iids_to_predict]

prediction=algo.test(testset)

prediction[5].est

pred_rating=[pred.est for pred in prediction]

i_max=np.argmax(pred_rating)

iid_reccomanded_most=iids_to_predict[i_max]




















