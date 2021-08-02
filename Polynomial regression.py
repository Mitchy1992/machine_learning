#Polynomial regression 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#import dataset and set x,y 
dataset = pd.read_csv('/Users/mitch/Downloads/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values #taking only the middle column , and all the rows 
y = dataset.iloc[:,-1].values #taking just the last row 

#linear regression model on dataset 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# perform polynomial regression then add onto the linear regression model 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # degree is set , to keep going to nth power to x .. b1x1 + b2x1^2 + b3x1^3 + b4x1^4 +...bnx1^n
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#plot linear regression 
plt.scatter(x,y,color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth/Bluff (LR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#plot polynomial regression 
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth/Bluff PR') 
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predict single value 
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))

