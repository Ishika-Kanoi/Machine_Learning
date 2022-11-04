# -*- coding: utf-8 -*-
"""
Salary Classification: Handling Categorical Data (One Hot Encoding)
Predict salary for x=['Development',1100,2,3]
"""

import pandas as pd
df = pd.read_csv('Salary_Classification.csv')
df.isnull().any(axis=0)

df.columns
#no missing data, separating features,labels
features = df.iloc[:,0:4].values
labels = df.iloc[:,4].values

df.dtypes
df['Department'].unique()

#problem: department is object
#we need to convert categorical data to numbers
#scheme: ONE HOT ENCODING
#info--> encoded()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# we wil use encoder, encoing scheme: One hot encoder,
# on what column = [o]), what about others: 
#remainder:'passthrough
cTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder = 'passthrough')
#give the transformer data to work upon
cTransformer.fit_transform(features)
#dtyep=object we need to change as its an object
import numpy as np
features = np.array(cTransformer.fit_transform(features),dtype = np.float32)
type(features)
features.dtype
#float32
#dropping the redundant column for a better ML model
#handling the DUMMY variable trap
features = features[:,1:]
#only 2 dummy variables are left


##train test split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size =0.2)

#FEATURE SCALING

#standard scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

features_train = sc.fit_transform(features_train)
#80%
features_test = sc.transform(features_test)



#building model
regressor = LinearRegression()
#model
regressor.fit(features_train,labels_train)
pred = regressor.predict(features_test)
pd.DataFrame(zip(pred,labels_test))

#calculating regressor score.- overfitting
regressor.score(features_train,labels_train)
regressor.score(features_test,labels_test)


#we were predicting salary
x=['Development',1100,2,3]
#error 1 d array
x = np.array(x)
regressor.predict(x)
#wrong again #reshape
x = x.reshape(1,4)
x.shape
#we reqire 5 dimension data 
#data is still categorical
#fit is not required
x = np.array(cTransformer.transform(x),dtype=np.float32)
#regressor.predict(x)
#mismatch

#data in 6d
#dropping the first column

x = x[:,1:]
regressor.predict(x)
#2897336.5 is the predicted salary
