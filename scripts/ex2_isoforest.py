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
best_n_estimators = 0

print('Starting grid search')
for n_estimators in range(100, 600, 100):
    print('Testing number of estimators : ' + str(n_estimators))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)

    isoforest = IsolationForest(n_estimators=n_estimators, verbose=1, warm_start=False)
    isoforest.fit(X_train)

    predictions = isoforest.predict(X_test)
    score = roc_auc_score(Y_test, predictions)
    print('Score : ' + str(score))
    print()

    if score > best_roc_auc_score:
        best_roc_auc_score = score
        best_n_estimators = n_estimators
        best_predictions = predictions

print('Grid search complete')
print('Best score : ' + str(best_roc_auc_score))
print('Best number of estimators : ' + str(best_n_estimators))

numpy.save('../data/isoforest-predictions.npy', best_predictions)
numpy.save('../data/isoforest-targets.npy', Y_test)
numpy.save('../data/isoforest-score.npy', best_roc_auc_score)
