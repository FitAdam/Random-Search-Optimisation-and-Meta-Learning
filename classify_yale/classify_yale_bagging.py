from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


print('Loading data...')

# load array
y = load('yaleExtB_target.npy')
X = load('yaleExtB_data.npy')

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


new_clf = BaggingClassifier(
          DecisionTreeClassifier(random_state=42), n_estimators=1000,
          max_samples=100, bootstrap=False, n_jobs=-1, random_state=42)
new_clf.fit(X_train, y_train)
y_pred = new_clf.predict(X_test)

print('The new accuracy score after Bagging and Decision Tree classifier is :')
print(accuracy_score(y_test, y_pred))

y_pred = new_clf.predict(X_test) # reocognises the test images 
print(classification_report(y_test, y_pred)) # the recognition accuracy
