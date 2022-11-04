# -*- coding: utf-8 -*-
"""
SimpleImputer is a scikit-learn class which is helpful in handling the missing data in the predictive model dataset.
It replaces the NaN values with a specified placeholder. 
It is implemented by the use of the SimpleImputer() method
"""
import pandas as pd
import numpy as np

df = pd.read_csv('cricket_salary_data.csv')

#we have age as missing data
df.shape
df.isnull().any(axis=0)
df['Age'].isnull().sum()
#missing value

#data should not have a missing data when we train
#(imputation handle missing data by replacing with mean)

features = df.iloc[:,0:3].values#name,salary
labels = df.iloc[:,3].values#grade

from sklearn.impute import SimpleImputer
from numpy import nan
imputer = SimpleImputer(missing_values= nan,strategy='mean')

features[:,1:2] = imputer.fit_transform(features[:,1:2])

#here the nan vals are replaced And the values are ready to be fed into the model
pd.isna(features)
