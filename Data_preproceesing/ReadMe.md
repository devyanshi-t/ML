# Data Pre-Processing
Data preprocessing is the first and very important step before we try and fit our data to a model.

# Steps:
## Importing essential libraries
 ```bash
 import numpy as np
 import matplotlib.pyplot as plt
 import pandas as pd
 ```
 ## Importing Data set
 Set up the working directory   with the folder which contains your csv file.<br/>
 Create a matrix for independent and dependent variables.
 ```bash
 dataset=pd.read_csv['Data.csv']
 X=dataset.iloc[:,:-1]
 Y=dataset.iloc[:3]
 ```
 Note: Index in python starts from 0
 ## Dealing with Missing Data in the dataset
 We might come across a case where values are missing in some columns in that case it is not advisable to ignore those rows as they might contain crucial data.<br/>
 So the solution to replace the empty value by the mean of the coulumn value( Other methods like replacing with median are also an option)<br/>
 ```bash 
 from sklearn.preprocessing import Imputer
 imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
 imputer=imputer.fit(X[:,1:3])
 X[:,1:3]=imputer.transform(X[:,1:3])
 ```
 Note: 1: 3 will correspond to col 1,2 as upperbound is excluded.<br/>
 ## Categorical Data
 Variables which contain categories needs to be encoded into numeric.
 method 1: by LabelEncoder
 ```bash
 from sklearn.preprocessing import LabelEncoder
 labelencoder_X=LabelEncoder()
 X[:,0]=labelencoder_X.fit_transorm(X[:,0])
 labelencoder_Y=LabelEncoder()
 y=labelencoder_Y.fit_transorm(y)
```
method 2: by OneHotEncoder
It is based on the concept of dummy variables where number of categories=number of dummy column introduced.
```bash
from sklearn.preprocess import OneHotEncoder
oh=OneHotEncoder(categorical_feature=[0])
X=oh.fit_transform(X).toarray()
```
## Spllting Data set into test set and training set
It is extremely important and is required to prevent over fitting,
``` bash
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=(X,Y,test_size=0.2) # 20% is test set
```
## Feature Scaling
 The columns need to be in similar scale.Can be acheived by Standarization or Normalization
 ``` bash 
 from sklearn.preprocessing import StandardScaler
 sc_X=StandardScaler()
 X_train=sc_X.fit_transorm(X_train)
 X_test=sc_X.transform(X_test)
 ```
 Note: We do not  need to apply feature scaling to dependent variable  usually in classification problems but it is required in regression problems.

 
