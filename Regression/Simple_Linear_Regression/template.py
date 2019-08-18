#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 23:26:15 2019

@author: devyanshitiwari
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,0]
Y=dataset.iloc[:,1]
# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting simple linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
X_train=X_train.values.reshape(-1,1)
X_test=X_test.values.reshape(-1,1)
regressor.fit(X_train,Y_train)
# checking predictions
y_pred=regressor.predict(X_test)

# r2 and mean squred error
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(Y_test,y_pred))
print( mean_squared_error(Y_test,y_pred))

#PLotting the results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

