#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 01:10:43 2019

@author: devyanshitiwari
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

# creating the regressor
from sklearn.ensemble import RandomForestRegressor
rg=RandomForestRegressor(n_estimators=300,random_state=0)
rg.fit(X,y)

# predictions
y_pred=rg.predict([[6.5]]) # 160333

# visualization
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, rg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()