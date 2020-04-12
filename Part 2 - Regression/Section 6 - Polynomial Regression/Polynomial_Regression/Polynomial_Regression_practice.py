#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:41:49 2020

@author: sahanaasokan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values #we are using only using column
y = dataset.iloc[:, 2].values

#from sklearn.model_selection import train_test_split
#xtrain,xtest,ytrain,ytest=train_test_split(x,y,testsize)

#Polynomial Regression is a form of linear regression
# in which the relationship between the independent 
# variable x and dependent variable y is modeled as 
# an nth degree polynomial

#Building Linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor= LinearRegression()
linear_regressor.fit(x,y)

#Building polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4) 
x_poly = poly_reg.fit_transform(x)
#matrix with features and 1 components,transformed matrix in polynomial terms

poly_reg.fit(x_poly,y)  # applying polynomial fit to x_poly/y

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y) #applying linear regression to our x in terms of polynomial


#Visualizing the linear regression results
plt.scatter(x,y,color='red') #orginal
plt.plot(x,linear_regressor.predict(x),color='blue') #Prediction results
plt.title('Truth ir Bluff (Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




#Visualizing the polynomail regression model
plt.scatter(x,y,color='red') #orginal
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue') #Prediction results
plt.title('Truth ir Bluff (Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




