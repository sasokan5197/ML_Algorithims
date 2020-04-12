#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Regressinons
# y = a+bx This is simple linear regression
#y= a+bx+b2x2+b3x3     This is multiple linear regression
#Multiple Linear Regression has multiple factors of what might affect y

#Assumptions of Linear Regression // you have to assume these assumptons are T

#Linearity
#Homescedasticity
#Multivariable normality
#Indepdendence of errors
#Lack of Multicolllinearity


# Profit = y 
# R/D Spend /Admin /Marketing/ State-categorical (Dummy Variables) Map it too different columsn
# Create a dataset for the dummy variables
# We cant include both California/New York
# What happens when we have more than 1....
#always omit ony dummmy variable


#what is the p-value
#How likely it is to get a result like this if the null hypothesis is true.

#trying to prove against h_0 
# trying to prove for h_1
# what is the significance value

#smaller the p value the more evidence you have the null hypothesis is wrong.
# greater the p value the less evidence...



# 5 methods of building models.
# All-in
# Backward Elimination
#Forward Selection
#Bidirection Elimination
#Score Comparison

"""
Created on Mon Mar 16 18:27:37 2020

@author: sahanaasokan
"""


import numpy as np #numpy is the library that has the mathematical code
import matplotlib.pyplot as plt #library for the plots/charts
import pandas as pd # this is the dataset for importing/handling datasets


dataset = pd.read_csv('50_Startups.csv')

x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values

#we have to encode the state variables...
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import  ColumnTransformer

ct= ColumnTransformer([('encoder',OneHotEncoder(),[3])], remainder = 'passthrough')
x= np.array(ct.fit_transform(x), dtype= np.float)



#Avoid the dummy variable trap
#x=x[:,1]

#Split the data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size= 0.20 )


#Step 2 Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression() #created object of the class
regressor.fit(xtrain,ytrain) #created regressor fitting it to xtrain/ytrain

#Step 3 Predicting Test Results
y_pred= regressor.predict(xtest) # y predicted values



#building the optimal model using backward elimination
import statsmodels.api as sm
x= np.append(arr= np.ones((50,1)).astype(int),values=x, axis =1,)
#axis = 0 line // axis = 1 column
#arr adding the 1's to the end that is what arr=x
# now we are adding x array to the column of 1's.
#why do we have to add 1??????

x_opt= x[:,[0,1,2,3,4,5]]
# optimal matrix of all the independent features
#matrix will only contain the variables with high impact on the profit
# we need to select significance level
#Fit the new model to x_optimal and y
regressor_ols= sm.OLS(y,x_opt).fit()
#Ordinary least Squares
regressor_ols.summary()

#Keep repreating last three lines of code till features p<t
