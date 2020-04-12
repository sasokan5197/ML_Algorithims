#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:33:26 2020

@author: sahanaasokan


purpose: categorize a random variable to a category.
Choose the number of K neighbhors- you choose this value
Take the K-nearest neighbhors of the new data point according to
the Euclidean distance
Among the K neighhors count the number of data points in each category
Assign the new data point to the category 
where you counted the most neighbhors

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
#y=y.reshape(-1,1)



#split the training set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2, random_state=0)


#Scale the data
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
xtrain=sc_x.fit_transform(xtrain)


#Split
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(xtrain, ytrain)

#Predicting Test Set Results
y_pred=classifier.predict(xtest)


#Making the fusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest, y_pred)

#Visualizing the training set results
from matplotlib.colors import ListedColormap
xset,yset=xtrain,ytrain

x1,x2= np.meshgrid(np.arange(start=xset[:,0].min()-1,stop=xset[:,0].max()+1,step=0.1),             
                   np.arange(start=xset[:,1].min()-1,stop=xset[:,1].max()+1,step=0.1))

plt.contour(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75,cmap = ListedColormap(('red','green')))
                                    
                                    
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset ==j,0],xset[yset ==j,1],
    c= ListedColormap(('red','green'))(i),label=j)



plt.title('KNN(Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

















