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

frame = pandas.read_pickle('../data/cicids2017.pkl')

for index, column in enumerate(frame.columns):
    frame.columns.values[index] = frame.columns.values[index].strip()

frame['Destination Port'] = frame['Destination Port'].apply(str)

frame = frame.dropna()

frame = frame.drop(['Flow Bytes/s', 'Flow Packets/s'], axis=1)

labels = pandas.DataFrame(frame['Label'].copy(), columns = ['Label'])

frame = frame.drop('Label', axis=1)

modified_labels_list = []

for label in labels.values:
    if label == 'BENIGN':
        modified_labels_list.append(0)
    else:
        modified_labels_list.append(1)

modified_labels = pandas.DataFrame(modified_labels_list, columns = ['Label'])

frame.to_pickle("../data/refined-cicids2017.pkl")
labels.to_pickle("../data/original-labels.pkl")
modified_labels.to_pickle("../data/simplified-labels.pkl")
