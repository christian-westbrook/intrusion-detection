# Data Management
import pandas

# External Interfaces
import glob
import kaggle
import os
from zipfile import ZipFile

# Evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

# Processing
import numpy
import scipy
from scipy.stats import chi2

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

X = pandas.read_pickle('../data/refined-cicids2017.pkl')
Y = pandas.read_pickle('../data/simplified-labels.pkl')

# Search k 5 through 25
best_predictions = []
best_roc_auc_score = 0
best_k = 0
best_d = 0

print('Starting grid search')
for k in range(5, 30, 5):
    print('Testing k : ' + str(k))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)

    pca = PCA()
    pca.fit(X_train)

    cumsum = numpy.cumsum(pca.explained_variance_ratio_)
    d = numpy.argmax(cumsum > 0.99) + 1
    print('Reduced dimensions to : ' + str(d))

    pca = PCA(n_components = d)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_reduced, Y_train.values.ravel())

    predictions = knn.predict(X_test_reduced)
    score = roc_auc_score(Y_test, predictions)
    print('Score : ' + str(score))
    print()

    if score > best_roc_auc_score:
        best_roc_auc_score = score
        best_k = k
        best_d = d
        best_predictions = predictions

print('Grid search complete')
print('Best score : ' + str(best_roc_auc_score))
print('Best k : ' + str(best_k))
print('Best d : ' + str(d))

numpy.save('../data/pca-knn-predictions.npy', best_predictions)
numpy.save('../data/pca-knn-targets.npy', Y_test)
numpy.save('../data/pca-knn-score.npy', best_roc_auc_score)
