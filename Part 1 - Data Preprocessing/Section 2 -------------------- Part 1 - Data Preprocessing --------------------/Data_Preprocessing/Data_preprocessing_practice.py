#How to import the libraries
import numpy as np #numpy is the library that has the mathematical code
import matplotlib.pyplot as plt #library for the plots/charts
import pandas as pd # this is the dataset for importing/handling datasets


#Importing the Data Set
dataset= pd.read_csv('Data.csv')
x= dataset.iloc[:, :-1].values # [x,y]= rows and columns. y-1= represents 4-3 columns etc.
y= dataset.iloc[:,3].values # returns all the rows for only column 3.
#y = dependent variable; x= independent variable
#missing Data
# we can replace the value of the column with the median/mean
  
#Cleaning Missing Data
# from sklearn.impute import SimpleImputer
# clean = SimpleImputer(missing_values='np.nan', strategy ='mean')
# use mean of columns using axis 0. axis 1 = rows. define object
# clean = clean.fit(x[:,1:3]) #fit the imputer on X on columns 1,3
# x[:,1:3]=SimpleImputer.transform(x[:,1:3])# transform X on columns 1,3 

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# create the object
encoder_X=LabelEncoder() #encodes categorical to numbers
x[:,0]=encoder_X.fit_transform(x[:,0]) #fit transformer is a method of the class labelencoder

#creates a map for categorical
#onehotencoder=OneHotEncoder(categorical_features = [0])
#x= onehotencoder.fit_transform(x).toarray()


#encoder_y=LabelEncoder()
#y=encoder_y.fit_transform(y)


#Splitting data into a training/ test set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2)
#splitting the data into test/train. Independent and dependent variable.
# The better he is trained(learn/understanding the data)= the better he can make better predictions.


#Feature Scaling

from sklearn.preprocessing import StandardScaler

#feature scaling= standardization, normalisation
scaler= StandardScaler()
xtrain=scaler.fit_transform(xtrain) # you have to fit object to training set and then transform it.
xtest= scaler.transform(xtest)


#what is the difference between fit and transform.



