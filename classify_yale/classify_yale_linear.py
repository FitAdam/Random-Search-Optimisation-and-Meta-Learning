"""
from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split # loads functions from the ML library sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

path = r'C:\Users\Vdvm\Documents\Projects\AI_project'
# load array
y = load('yaleExtB_target.npy')
X = load('yaleExtB_data.npy')

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print('The new accuracy score is :')
print(accuracy_score(y_test, predicted.round(), normalize=False))

"""