
# Random Forest Regression

The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. <br/>
## Steps
1. Pick a random K data point from the training set.<br/>
2.  Build the decision trees associated to these K data points.<br/>
3.  Choose the number of  trees you want to build and repeat step 1 and  2.<br/>
4. For a new data point let all the decision trees predict a value of y and then take average of all th predicted values.

## Problem Statement
An employee joining a new company tell that he was a 6.5 level emplyee in his former company and and had a salary of 160K.The HR however is not conviced and decided to check whether the person is honest or not.

## Solution
Salary of employees based on their levels is predicted by  the  random forest regrerssion <br/>

## Result

<p align="center">
<img src="./1.png"></br>


Since the model predicts the salary to be  160333means employee was honest.


