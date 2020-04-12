#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:29:05 2020

@author: sahanaasokan
"""


#Classification Trees
#Regression Trees


#Scatter plot willl be split up into segments
#multiple splits
#information entropy
#algorithim finds optimal splits from the dataset
#the splits are called terminal leaves

#regression trees makes splits 
#regression trees are nonlinear and non continous
#take the average of each of the terminal leaf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values




# Feature Scaling
#from sklearn.preprocessing import StandardScaler

"""sc_x = StandardScaler()
sc_y = StandardScaler()
x= sc_x.fit_transform(x)
y=sc_y.fit_transform(y)"""



#Fitting DecisionTreeRegressor to the dataset

from sklearn.tree import DecisionTreeRegressor
decision_regressor=DecisionTreeRegressor(random_state=0)
decision_regressor.fit(x,y)

y_pred=decision_regressor.predict(x)


#We have to represent this model with more values because it is
#non continous model.
#Visualization with higher resolution and smoother curves
x_grid = np.arange(min(x),max(x),0.001)# gives more values of x
x_grid= x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,decision_regressor.predict(x_grid), color='blue')
plt.title('truth or bluff Decision Tree Regression')
plt.show()

#range of independent variables are split. the vertical lines show
#where the different intervals are
#each level is a value.



