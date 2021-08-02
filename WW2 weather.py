import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as seabornInstance 


dataset = pd.read_csv('/Users/mitch/Downloads/Weather_WW2/Summary of Weather.csv')
print(dataset.shape) # get total rows and column count
print(dataset.describe()) # statistical details of dataset  


#Plot a graph of min temp vs max temp
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('Mintemp vs Maxtemp')
plt.xlabel('Mintemp')
plt.ylabel('Maxtemp')
plt.show() 


seabornInstance.displot(dataset['MaxTemp'], kde='true', bins=60)
plt.title('Avg MaxTemp')
plt.show()


#dividing data into attributes and labels , x = independent y = dependent 
x = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1) 

#split data into training and test set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train) #training the algorithm 

#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)

#predict test results 
y_pred = regressor.predict(x_test)

#compare the actual output values for X_test with the predicted values: y_test = actual result, y_pred = predicted result
dataframe = pd.DataFrame({'Actual':y_test.flatten(), 'Predicted':y_pred.flatten()}) 

#plot result comparision bar graph 
df1 = dataframe.head(26)
df1.plot(kind='bar',figsize=(8,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#plot prediction vs actual graph 
plt.scatter(x_test, y_test, color='grey')
plt.plot(x_test, y_pred, color= 'red', linewidth =2)
plt.title('Actual vs Predicted')
plt.show()



#perfrom evaluation metric using the sklearn.metrics library
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))







