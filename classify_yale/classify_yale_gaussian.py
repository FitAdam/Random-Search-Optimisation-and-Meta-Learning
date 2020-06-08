from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split # loads functions from the ML library sklearn 
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

path = r'C:\Users\Vdvm\Documents\Projects\AI_project'
# load array
y = load('yaleExtB_target.npy')
X = load('yaleExtB_data.npy')

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian Naive Bayes Classification
gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)

#print("Number of mislabeled points out of a total %d points : %d"
      #  % (X_test.shape[0], (y_test != y_pred).sum()))

#print('The new accuracy score is :')
#print(accuracy_score(y_test, y_pred))


y_pred = gnb.predict(X_test) # reocognises the test images 
print(classification_report(y_test, y_pred)) # the recognition accuracy
