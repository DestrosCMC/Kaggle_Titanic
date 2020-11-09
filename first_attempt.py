# First attempt, accuracy of 61%, ranking of around 16,000

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col = 'PassengerId')
data = data.drop(columns = ['Name', 'Ticket', 'Cabin'])
#print(data.head())
#sns.heatmap(data.corr())
#plt.show()
data = data.fillna(data.mean())
Y = data.iloc[:,0]
X = data.iloc[:, 1:]
#print(X)
#print(Y)
#print(X.dtypes)
X = pd.get_dummies(X)
#print(data.shape)
#print(Y)

#missing_val_count_by_column = (data.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column > 0])
'''
for k in range(1,15):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    knnr = KNeighborsClassifier(n_neighbors=k, p=2)
    knnr.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    y_pred = knnr.predict(X_test)

    a = mean_squared_error(y_test, y_pred)
    print("The MSE is:", a, k)
'''

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=10, p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

a = mean_squared_error(y_test, y_pred)
print("The MSE is:",a)

#print(X.columns)
#knnr.fit(X, Y)

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col = 'PassengerId')
index = pd.DataFrame
test_data = test_data.fillna(test_data.mean())
test_data = test_data.drop(columns = ['Name', 'Ticket', 'Cabin'])
test_data = pd.get_dummies(test_data)
#print(test_data.columns)
#test_data_x = 

titanic_pred = knn.predict(test_data)
#print(titanic_pred)
#print(titanic_pred.shape)

index = pd.read_csv('/kaggle/input/titanic/test.csv')
index = index['PassengerId']
#print(index)
titanic_pred = pd.DataFrame(titanic_pred, columns = ['Survived'])
#print(titanic_pred)
output = pd.concat([index,titanic_pred], axis = 1)
#output = output.set_index('PassengerId')
#output = pd.DataFrame(test_data, titanic_pred)
#print(output)
output.to_csv('submission.csv', index = False)
