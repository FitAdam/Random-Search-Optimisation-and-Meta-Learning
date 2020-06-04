from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

print('Loading data...')
path = r'C:\Users\Vdvm\Documents\Projects\ML_project'
# load array
y = load('yaleExtB_target.npy')
X = load('yaleExtB_data.npy')

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

adaboost_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)

#train the model
print('Training the model...')

adaboost_clf.fit(X_train, y_train)

y_pred = adaboost_clf.predict(X_test)
print('The new accuracy score after AdaBoost and Decision tree classifier is :')
print(accuracy_score(y_test, y_pred))

new_clf = BaggingClassifier(
          DecisionTreeClassifier(random_state=42), n_estimators=1000,
          max_samples=100, bootstrap=False, n_jobs=-1, random_state=42)
new_clf.fit(X_train, y_train)
y_pred = new_clf.predict(X_test)

print('The new accuracy score after Bagging and Decision Tree classifier is :')
print(accuracy_score(y_test, y_pred))