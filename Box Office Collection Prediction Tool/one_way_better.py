# -*- coding: utf-8 -*-
"""
Box Office Collection Prediction Tool

Dataset: Bahubali2vsDangal.csv
  
    It contains Data of Day wise collections of the bollywood movies 
    Bahubali 2 and Dangal (in crores) for the first 9 days.    

Problem Statement:
    Develop a machine learning model based on linear regressor 
    to predict which movie would collect more money on the 10th day.


"""

import pandas as pd


df = pd.read_csv('Box_Office.csv')
df.info()
#splitting featues and labels
features = df.iloc[:,0:1].values
labels = df.iloc[:,1:3].values

#building the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting the model: because of small dataset we did
#                   not split the data into train & test

regressor.fit(features,labels)
pred = regressor.predict([[10]])

#collected the collections in two variables
bahubali_collection, dangal_collection = pred[0]

day=10
if bahubali_collection> dangal_collection:
    print("Bahubali2 will earn more on {0} day".format(day))
else:
    print("Dangal will earn more on {0} day".format(day))
    
