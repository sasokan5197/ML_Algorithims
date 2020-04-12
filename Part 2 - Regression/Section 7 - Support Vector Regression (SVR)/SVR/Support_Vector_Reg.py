#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:34:00 2020

@author: sahanaasokan
"""


#supports linear and nonlinear regression
#svr tries to fit as many options while limiting margin violations
#width of the street is controlled by epsilon

#SVR performs linear regression in a higher dimensional space
#fit as many instances as possible( multiple lines)
#epsilon controls width of the street.

#very good method to use.


#what are the requirements
#It requires a training set whcih covers domaon of interest.

#Collect training set
#Choose a kernel( Gaussian Regularization Noise)
#Form a correlation
#train your machine
#use those coefficients


# Ka=y
# K is the correlation matrix
# y is the vectir of values corresponding to K
# we are solving for a (unknown values)
# a = (k^-1)*y

#different regression - predictionerrors do not exceed threshold
#linear regression - minimize predction errors


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values
y= y.reshape(-1,1)
x=x.reshape(-1,1)



# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
x= sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


#Fitting SVR to the dataset
from sklearn.svm import SVR #class for svr regression
svr_reg= SVR(kernel='rbf')
svr_reg.fit(x,y)

y_pred=svr_reg.predict(x)

plt.scatter(x,y,color='green')
plt.plot(x,svr_reg.predict(x), color= 'pink')
plt.title('Truth or Bluff SVR')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()



















