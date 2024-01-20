#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 08:53:27 2022

@author: dai
"""

import pandas as pd
import numpy as np
import surprise

ratings = pd.read_csv("../Cases/ml-100k/u.data",sep='\t',names= ['uid', 'iid', 'rating','Date'])
ratings.head()

ratings.drop('Date',axis=1,inplace=True)

lowest_rating = ratings['rating'].min()
highest_rating = ratings['rating'].max()
print("Ratings range between {0} and {1}".format(lowest_rating,highest_rating))

###################### Converting the data into similarities options
reader = surprise.Reader(rating_scale = (lowest_rating,highest_rating))
data = surprise.Dataset.load_from_df(ratings,reader)

################################## Similarity options
similarities_options = {"name":"cosine" , "user_based":True}

# Default k = 40
algo = surprise.KNNBasic(sim_options=similarities_options)
output = algo.fit(data.build_full_trainset())

# Computing the cosine similarity matrix...
#Done computing similarity matrix
# The above .fit() calculates expected rating for all the users

##################################
# As we want expected rating of users=50 for items 217
pred = algo.predict(uid='50', iid='217')
score = pred.est
print(score)

iids = ratings["iid"].unique()
rec_50 = ratings[ratings["uid"]==50]
iids50 = rec_50["iid"]
print("List of iid that uid={0} has rated".format(50))
print(iids50)

iids_to_predict = np.setdiff1d(iids, iids50)
print("List of iid which uid={0} did not rate(in all {1}) :".format(50,len(iids_to_predict)))
print(iids_to_predict)

# rating arbitary set to 0
testset = [[50,iid,0.] for iid in iids_to_predict]
predictions = algo.test(testset)
predictions[5]

pred_ratings = np.array([pred.est for pred in predictions])

# Finding the index of maximum predicted rating
i_max = pred_ratings.argmax()

#Recommending the item with maximum predicted rating
iid_recommend_most = iids_to_predict[i_max] 
print("Top item to be recommended for user {0} is {1} with predicted rating as {2}".format(50,iid_recommend_most,pred_ratings[i_max]))















