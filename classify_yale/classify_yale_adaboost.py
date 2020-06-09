from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



print('Loading data...')

# load array
y = load('yaleExtB_target.npy')
X = load('yaleExtB_data.npy')

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# PCA 
nof_prin_components = 1200  # PARAMETER for optimisation in expereiments
pca = PCA(n_components=nof_prin_components, whiten=True).fit(X_train)
# applies PCA to the train and test images to calculate the principal components
X_train_pca = pca.transform(X_train) 
X_test_pca = pca.transform(X_test)

# train a neural network
adaboost_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=5000,
    algorithm="SAMME.R", learning_rate=0.2, random_state=42,)

#train the model
print('Training the model...')

adaboost_clf.fit(X_train_pca, y_train)

y_pred = adaboost_clf.predict(X_test_pca)
print(classification_report(y_test, y_pred)) # the recognition accuracy