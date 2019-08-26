#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:54:01 2019

@author: devyanshitiwari
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,4]

# Dealing with categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()
le.fit(X.iloc[:,3])
X.iloc[:,3]=le.fit_transform(X.iloc[:,3])
oh=OneHotEncoder(categorical_features=[3])
X=oh.fit_transform(X).toarray()

# Getting rid of the Dummy Variable Trap
X=X[:,1:]
# Splitting data into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting multiple Linear Regression into our model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test results
y_pred=regressor.predict(X_test)

#Accuracy
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(Y_test,y_pred))
print( mean_squared_error(Y_test,y_pred))

#Building the optimal model by Backward Elimation
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) # Adding the intercept
# chosen significance level(sl) =0.05
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

# removing x2
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#removing x1
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#removing 
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

# p value for all is less than sl so finish

# automatic backward eminationation using a loop instead

   
    