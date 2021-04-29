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

X = pandas.read_pickle('../data/refined-cicids2017.pkl').drop(['Destination Port'], axis=1)
Y = pandas.read_pickle('../data/simplified-labels.pkl')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)

covariance = numpy.cov(X_train, rowvar=0)
inverse_covariance = numpy.linalg.pinv(covariance)
centerpoint = numpy.mean(X_train, axis=0)

distances = []

for index, record in X_test.iterrows():
    p1 = record
    p2 = centerpoint
    distance = (p1 - p2).T.dot(inverse_covariance).dot(p1 - p2)
    distances.append(distance)

distances = numpy.array(distances)

# Search cutoff 0.70 through 0.95
best_predictions = []
best_roc_auc_score = 0
best_cutoff_threshold = 0

print('Starting grid search')
for cutoff_threshold in range(70, 100, 5):
    print('Testing threshold : ' + str(cutoff_threshold / 100))
    cutoff = scipy.stats.chi2.ppf((cutoff_threshold / 100), X_test.shape[1])
    prediction_indices = numpy.where(distances > cutoff)[0]

    predictions = []
    for index in range(0, len(X_test)):
        if index in prediction_indices:
            predictions.append(1)
        else:
            predictions.append(0)

    score = roc_auc_score(Y_test, predictions)
    print('Score : ' + str(score))
    print()

    if score > best_roc_auc_score:
        best_roc_auc_score = score
        best_cutoff_threshold = cutoff_threshold
        best_predictions = predictions

print('Grid search complete')
print('Best score : ' + str(best_roc_auc_score))
print('Best cutoff threshold : ' + str(best_cutoff_threshold))

numpy.save('../data/mahalanobis-predictions.npy', best_predictions)
numpy.save('../data/mahalanobis-targets.npy', Y_test)
numpy.save('../data/mahalanobis-score.npy', best_roc_auc_score)
