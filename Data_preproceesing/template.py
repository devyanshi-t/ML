#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:03:46 2019

@author: devyanshitiwari
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing data set
dataset=pd.read_csv['Data.csv'];
X=dataset.iloc[:,:-1] # independent variable
Y=dataset.iloc[:,3] # dependent variable

# missing data
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='Nan',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_Y=LabelEncoder()
Y=label_encoder_Y.fit_transform(Y)

oh=OneHotEncoder(categorical_feautre=[0])
X=oh.fit_transorm(X).toarray()

# Splitting The Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# Feature Scaling
from sklearn.preproceesing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

