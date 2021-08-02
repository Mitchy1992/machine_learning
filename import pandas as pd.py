#Import all the libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#Import dataset using panda
dataset = pd.read_csv('/Users/mitch/Downloads/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv') 
#Split the dependent and independent variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 

#Split the dataset into Training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 ) 

#Perform linear regression model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict test results 
y_pred = regressor.predict(X_test)

#Visualise the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#Visualise the testing set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 









