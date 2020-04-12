#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:45:17 2020

@author: sahanaasokan
"""
#y = b+bx
#y = dependent variable and x is the indepdent variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')


#simplelinear regression step 1(pre-processing)
# ordinary least squares method - best fitting line 
x= dataset.iloc[:,:-1].values # calls the indepedent variables
y=dataset.iloc[:,1].values #calls the dependent variable that we will test on


#we dont need to clean missing data because nothing is missing

#split data into training and test model

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size= 0.20 )
#model will be training based on x train and y train, then we will predict.



#Simple linear regression Step 2(traiing the modell)
# Fitting Simple Linear Regression to the training set.

from sklearn.linear_model import LinearRegression
regression_test= LinearRegression()
regression_test.fit(xtrain,ytrain)  #fits regressor object 
#training the train sets with linearreg model

#Step 3 (Predicting the Test Results)
ypred = regression_test.predict(xtest) #predicting the test set results

#Step 4 Visualizing the Training Set results
ypred_train= regression_test.predict(xtrain)
plt.scatter(xtrain,ytrain, color='red') # these are our real values
plt.plot(xtrain, ypred_train, color='blue') #predicted line 
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Step 5 Visualizing the Test Results
ypred_train= regression_test.predict(xtrain)
plt.scatter(xtest,ytest, color='red') # these are our real values
plt.plot(xtrain, ypred_train, color='blue') #predicted line 
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
