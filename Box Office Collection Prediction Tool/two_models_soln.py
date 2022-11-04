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
df.shape
df.isnull().any(axis=0)

bahubali = df[['Days','Bahu_collections']]
dangal =df[['Days','Dangal_collections']]

bahu_feat = bahubali.iloc[:,0:1]
bahu_lab= bahubali.iloc[:,1:2]

dangal_feat = dangal.iloc[:,0:1]
dangal_lab= dangal.iloc[:,1:2]

#splitting the data
from sklearn.model_selection import train_test_split
Bf_train,Bf_test,Bl_train,Bl_test = train_test_split(bahu_feat,bahu_lab,test_size=0.2,random_state=45)
Df_train,Df_test,Dl_train,Dl_test = train_test_split(dangal_feat,dangal_lab,test_size=0.2,random_state=45)

#building the model
from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()
regressor1.fit(Bf_train,Bl_train)

regressor2 = LinearRegression()
regressor2.fit(Df_train,Dl_train)


#predicting the values
bahucol =regressor1.predict([[10]])
dangcol =regressor2.predict([[10]])

if bahucol>dangcol :
    print("Bahubli has higher colection on 10th day")
else:
    print("Dangal has higher collection on 10th day")


#Calculatiing Simple Test Scores   
regressor1.score(Bf_test,Bl_test)

regressor2.score(Df_test,Dl_test)
