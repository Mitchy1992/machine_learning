#Import all the Libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Import dataset 
dataset = pd.read_csv('/Users/mitch/Downloads/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv') 
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values 




#Encode the categorical data into numerical vector matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x)) 




# Split data into training set and test set
from sklearn.model_selection import train_test_split 
x_train , x_test, y_train, y_test= train_test_split(x, y, test_size = 0.2, random_state = 0)  

 

#Train the model using training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train, y_train)



#Predicting the test results 
y_pred = regressor.predict(x_test)
#Display numerical value , upto 2 decimal places
np.set_printoptions(precision =2)
#Display vectors of real profits vs predicted profits
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)) 



 







