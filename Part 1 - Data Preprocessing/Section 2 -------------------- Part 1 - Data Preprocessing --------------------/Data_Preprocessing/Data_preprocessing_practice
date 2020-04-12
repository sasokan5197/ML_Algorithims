#How to import the libraries
import numpy as np #numpy is the library that has the mathematical code
import matplotlib.pyplot as plt #library for the plots/charts
import pandas as pd # this is the dataset for importing/handling datasets


#Importing the Data Set
dataset=pd.read_csv('Data.csv')
x= dataset.iloc[:, :-1].values # [x,y]= rows and columns. y-1= represents 4-3 columns etc.
y= dataset.iloc[:,3].values # returns all the rows for only column 3.
#missing Data
# we can replace the value of the column with the median/mean
  #preprocess using Imputer
from sklearn.impute import SimpleImputer
clean = SimpleImputer(missing_values='NaN', strategy ='mean')
 # use mean of columns using axis 0. axis 1 = rows. define object

clean = clean.fit(x[:,1:3]) #apply clean fit to X on columns 1,3

x[:,1:3]=SimpleImputer.transform(x[:,1:3])#apply transform to X on columns 1,3 



