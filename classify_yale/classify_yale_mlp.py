from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split # loads functions from the ML library sklearn 
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

path = r'C:\Users\Vdvm\Documents\Projects\AI_project'
# load array
y = load('yaleExtB_target.npy')
X = load('yaleExtB_data.npy')

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# PCA 
nof_prin_components = 200  # PARAMETER for optimisation in expereiments
pca = PCA(n_components=nof_prin_components, whiten=True).fit(X_train)
# applies PCA to the train and test images to calculate the principal components
X_train_pca = pca.transform(X_train) 
X_test_pca = pca.transform(X_test)


# train a neural network
nohn = 200 # nof hidden neurons
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(nohn,), solver='adam', activation='tanh', batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca) # reocognises the test images 
print(classification_report(y_test, y_pred)) # the recognition accuracy