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

# ------------------------------------------------------------------------------
# Function : retrieve_combine_and_pickle()
# Engineer : Christian Westbrook
# Abstract : This function begins by defining all of the CSV files that are
#            expected from the dataset. Each CSV is loaded into a pandas
#            dataframe and then appended to a list of dataframes. Once all CSV
#            files have been loaded into dataframes, the dataframes are merged
#            into a single large dataframe representing the dataset. This final
#            dataframe is then written to disk in pickle format.
# ------------------------------------------------------------------------------
def retrieve_combine_and_pickle():
    # Check if a root /data directory exists, and create it if it doesn't
    if not os.path.exists("../data/"):
        os.makedirs("../data")

    # Retrieve the dataset in .zip archive format
    !kaggle datasets download cicdataset/cicids2017 -q

    # Move the dataset into the root /data directory.
    #!mv cicids2017.zip ../data/
    os.replace("./cicids2017.zip", "../data/cicids2017.zip")

    # Unzip the dataset in place
    with ZipFile('../data/cicids2017.zip', 'r') as zipObj:
       zipObj.extractall(path="../data/")

    # Grab all CSV file paths in the root /data directory
    file_paths = glob.glob("../data/**/*.csv", recursive=True)

    # Move all CSV files from the unzipped folder structure into the root /data directory
    for index, path in enumerate(file_paths):
        os.replace(path, "../data/" + path.split("\\")[len(path.split("\\")) - 1])

    # Grab all CSV file paths in the root /data directory
    file_paths = glob.glob("../data/**/*.csv", recursive=True)

    # Read each CSV into a pandas dataframe
    frames = []
    for index, path in enumerate(file_paths):
        frames.append(pandas.read_csv(path))

    # Merge dataframes vertically
    combined_frame = pandas.concat(frames, axis=0)

    # Reset row indices
    combined_frame = combined_frame.reset_index(drop=True)

    # Write combined dataframe to disk
    combined_frame.to_pickle("../data/cicids2017.pkl")

    # Clean up the root /data directory
    for index, path in enumerate(file_paths):
        os.remove(path)
    os.rmdir("../data/MachineLearningCSV/MachineLearningCVE/")
    os.rmdir("../data/MachineLearningCSV/")
    os.remove("../data/cicids2017.zip")
    os.remove("../data/MachineLearningCSV.md5")

retrieve_combine_and_pickle()
