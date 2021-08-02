
#import all the libararies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#import dataset and create independent and dependent variable 
dataset = pd.read_csv('/Users/mitch/Desktop/position_salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values 

y = y.reshape(len(y), 1) # to covert into vertical matrcies , 2D array [[]]


#scale the dataset (making all data uniform)
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() #need to call standard scaler for x,y beacause data scale is vastly different for both variables 
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y) 


#perfrom svr on dataset
from sklearn.svm import SVR 
regressor = SVR(kernel ='rbf')  
regressor.fit(X,y) 
 
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))) #using transform and inverse_transform method to convert the feature scaled values into the normal values
print(y_pred)



# plot the graph 
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(X,y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('SVR')
plt.xlabel('position')
plt.ylabel('salaries')
plt.show() 







