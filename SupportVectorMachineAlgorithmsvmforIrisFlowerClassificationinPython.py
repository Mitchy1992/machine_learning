import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv('iris.data')
feature_cols=['sepal_len', 'sepal_wid']
X=df[feature_cols]
y=df['class']
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
clf=svm.SVC(kernel='linear',decision_function_shape='ovr')
clf.fit(X_train,y_train)
pred_y=clf.predict(X_test)
x_min, x_max = X['sepal_len'].min() - 1, X['sepal_len'].max() + 1
y_min, y_max = X['sepal_wid'].min() - 1, X['sepal_wid'].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
#plt.subplot(1, 1, 1)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X['sepal_len'], X['sepal_wid'], c=y, cmap=plt.cm.Paired,edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
#plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
print("True values :" ,pred_y[:10])
print("Pred values :" ,y_test[:10])
print("R^2 or Accuracy :" ,clf.score(X_test,y_test))