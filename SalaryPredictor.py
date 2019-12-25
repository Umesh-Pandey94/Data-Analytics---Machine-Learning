# -*- coding: utf-8 -*-
"""
Title: Salary Predictor using Simple Linear Regression Algorithm
@author: Umesh Pandey
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import the DataSet
df=pd.read_csv('SalaryVsExperience.csv')

#Check the dimensions and shape
size=df.size
row_col=df.shape
dimension=df.ndim

print("Size of the dataset is" ,size, "\n Shape of the dataset is",row_col," \n Dimension of the dataset is",dimension) 


#Creating matrices and vectors
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1:].values

#Data Pre-Processing
from sklearn.model_selection._split import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=0)

#2-D array transformation
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)

#Applying Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)

Y_train_predict=lr.predict(X_train)
Y_test_predict=lr.predict(X_test)

#Training Set and visualisation 
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,Y_train_predict,color="blue")
plt.title(label="Training Set Results-Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Test Set and visualisation 
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,lr.predict(X_test),color="blue")
plt.title(label="Test Set Results-Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()












