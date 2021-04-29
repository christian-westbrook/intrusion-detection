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

# Experiment 1
mahalanobis_predictions = numpy.load('../data/mahalanobis-predictions.npy')
mahalanobis_targets = numpy.load('../data/mahalanobis-targets.npy')

# Experiment 2
isoforest_predictions = numpy.load('../data/isoforest-predictions.npy')
isoforest_targets = numpy.load('../data/isoforest-targets.npy')

# Experiment 3
regression_predictions = numpy.load('../data/multiple-linear-regression-predictions.npy')
regression_targets = numpy.load('../data/multiple-linear-regression-targets.npy')

# Experiment 4
pca_knn_predictions = numpy.load('../data/pca-knn-predictions.npy')
pca_knn_targets = numpy.load('../data/pca-knn-targets.npy')

# Experiment 5
pca_lof_predictions = numpy.load('../data/pca-lof-predictions.npy')
pca_lof_targets = numpy.load('../data/pca-lof-targets.npy')

for index in range(0, len(isoforest_predictions)):
    if isoforest_predictions[index] == -1:
        isoforest_predictions[index] = 0

for index in range(0, len(regression_predictions)):
    if regression_predictions[index] >= 0.5:
        regression_predictions[index] = 1
    else:
        regression_predictions[index] = 0

for index in range(0, len(pca_lof_predictions)):
    if pca_lof_predictions[index] == -1:
        pca_lof_predictions[index] = 0

mahalanobis_recall = recall_score(mahalanobis_targets, mahalanobis_predictions)
isoforest_recall = recall_score(isoforest_targets, isoforest_predictions)
regression_recall = recall_score(regression_targets, regression_predictions)
pca_knn_recall = recall_score(pca_knn_targets, pca_knn_predictions)
pca_lof_recall = recall_score(pca_lof_targets, pca_lof_predictions)

mahalanobis_precision = precision_score(mahalanobis_targets, mahalanobis_predictions)
isoforest_precision = precision_score(isoforest_targets, isoforest_predictions)
regression_precision = precision_score(regression_targets, regression_predictions)
pca_knn_precision = precision_score(pca_knn_targets, pca_knn_predictions)
pca_lof_precision = precision_score(pca_lof_targets, pca_lof_predictions)

mahalanobis_auroc = roc_auc_score(mahalanobis_targets, mahalanobis_predictions)
isoforest_auroc = roc_auc_score(isoforest_targets, isoforest_predictions)
regression_auroc = roc_auc_score(regression_targets, regression_predictions)
pca_knn_auroc = roc_auc_score(pca_knn_targets, pca_knn_predictions)
pca_lof_auroc = roc_auc_score(pca_lof_targets, pca_lof_predictions)

name_series = pandas.Series(['Mahalanobis Distances', 'Isolation Forests', 'Multiple Linear Regression', 'PCA K-Nearest Neighbors', 'PCA Local Outlier Factor'])
precision_series  = pandas.Series([mahalanobis_precision, isoforest_precision, regression_precision, pca_knn_precision, pca_lof_precision ])
recall_series  = pandas.Series([mahalanobis_recall, isoforest_recall, regression_recall, pca_knn_recall, pca_lof_recall ])
auroc_series = pandas.Series([mahalanobis_auroc, isoforest_auroc, regression_auroc, pca_knn_auroc, pca_lof_auroc ])

metrics_frame = pandas.DataFrame({'Method' : name_series, 'Precision' : precision_series, 'Recall' : recall_series, 'AUROC' : auroc_series })
print(metrics_frame)
