#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:20:03 2019

@author: devyanshitiwari
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:2] #  1:2 so that we could make it a matrix not a vector
y=dataset.iloc[:,2]

# Not splitting into test set as we need max info possible for predicitions 

# Fitting to linear regression model
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X,y)

# Fitting to polynomial regression model

from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=4)
X_poly=pr.fit_transform(X)
linear_reg2=LinearRegression()
linear_reg2.fit(X_poly,y)

# Visualising the Linear Regression 
plt.scatter(X,y,color='violet')
plt.plot(X,linear_reg.predict(X),color='red')
plt.title('Truth or False')
plt.xlabel('Position held')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression
plt.scatter(X,y,color='green')
plt.plot(X,linear_reg2.predict(pr.fit_transform(X)),color='red')
plt.title('Truth or False')
plt.xlabel('Position held')
plt.ylabel('Salary')
plt.show()

# Concept of Xgrid

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)