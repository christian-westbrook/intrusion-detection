#!/usr/bin/env python
# coding: utf-8

# # Applications of Anomaly Detection Techniques to Intrusion Detection

# ## Before You Begin
# 
# This section contains important information about executing the code in this notebook along with an alternative way to access the source code. If you don't plan on executing any code yourself, feel free to skip this section.
# 
# Many of the models used in this work were generated using the sklearn package. This package does not include support for processing with GPUs. This, combined with a large dataset, means that some of the models generated in this notebook can take a significant amount of time to build. This is something to keep in mind if you are planning to run these experiments yourself.
# 
# Some of the source code in this notebook makes assumptions about where you are running it. If you are planning on executing the code in this notebook, ensure that you are either running it within a clone of the [associated repository](https://github.com/christian-westbrook/intrusion-detection) at its default location, or that you understand how to adapt relative file paths to meet your needs. A good way to determine if a block of code makes this assumption is to look for instances where the `os` module is being used. If you aren't sure, feel free to clone a fresh copy of the repository and run your new copy of the notebook at its default location.
# 
# This notebook contains more than just source code. If you're only interested in using the source code, you may prefer to use our scripts instead of this notebook. These scripts are located in the root `/scripts` directory. For instructions on how to use them, refer to the README located in the root directory of the repository.

# ## Imports
# 
# - *pandas* for processing and rendering tabular data
# - *glob* for enabling dynamic interfacing with the file system
# - *kaggle* for interfacing with [Kaggle](https://www.kaggle.com/)
# - *os* for interfacing with the machine where this notebook is being ran
# - *zipfile* for managing .zip archives
# - *sklearn* for modeling and evaluation metrics
# - *numpy* for vector and matrix processing
# - *scipy* for probabilistic processing

# In[1]:


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


# ## Introduction
# 
# The goal of this work is to explore how existing anomaly detection techniques can be applied to the domain of intrusion detection. To this end we perform a series of experiments testing well-known anomaly detection techniques against the CICIDS2017 dataset of network intrusion events.
# 
# We begin by exploring the given dataset and preparing its data to serve as input to our models. We then continue with a series of experiments, each demonstrating the effectiveness of a particular anomaly detection technique for the task of classifying whether network traffic events were benign or malicious. These experiments can be considered binary classification tasks with imabalanced classes. Our evaluation metric for each technique is the area under the receiver operating characteristic curve.

# ## Defining Terms

# ***AUROC*** - Area under the receiver operating characteristic curve  
# ***CICIDS2017*** - A dataset of simulated packet capture events containing both benign and attack events  
# ***Mahalanobis distance*** - A measure of the distance between a point and a distribution  
# ***ROC*** - Receiver operating characteristic curve representing the true positive rate plotted against the false positive rate at all classification thresholds

# ## Loading Data

# In this section we describe how to retrieve the dataset used in this notebook. There is more than one way to retreive the dataset. Use either of the following methods to retrieve the dataset and prepare it for use in the rest of the notebook.

# ### Method 1: Manual Retrieval from Kaggle

# #### Requirements
# 
# You need to have an account with [Kaggle](https://www.kaggle.com/). The dataset is retrievable from [a hosted location on Kaggle](https://www.kaggle.com/cicdataset/cicids2017). To prepare the data for use in this notebook, start by navigating to the dataset on Kaggle using the following link and downloading the dataset.
# 
# https://www.kaggle.com/cicdataset/cicids2017
# 
# Extract the contents of this .zip archive file into the root `/data` directory. Explore the contents of the extracted archive and move all .csv files from their sub-directories into the root `/data` directory. The following script will expect that all .csv files from the dataset are located in the root `/data` directory. Use the `combine_and_pickle()` script to merge the separate CSV files into a single pickled dataset for a simplified processing experience.

# #### Automated Combination and Pickling

# In[ ]:


# ------------------------------------------------------------------------------
# Function : combine_and_pickle()
# Engineer : Christian Westbrook
# Abstract : This function begins by defining all of the CSV files that are
#            expected from the dataset. Each CSV is loaded into a pandas
#            dataframe and then appended to a list of dataframes. Once all CSV
#            files have been loaded into dataframes, the dataframes are merged
#            into a single large dataframe representing the dataset. This final
#            dataframe is then written to disk in pickle format.
# ------------------------------------------------------------------------------
def combine_and_pickle():
    # Grab all CSV file paths in the root /data directory
    file_paths = glob.glob("../data/**/*.csv", recursive=True)

    # Read each CSV into a pandas dataframe
    frames = []
    for index, path in enumerate(file_paths):
        frames.append(pandas.read_csv(path))

    # Merge dataframes vertically
    combined_frame = pandas.concat(frames, axis=0)
    
    # Reset row indices
    combined_frame.reset_index(drop=True)

    # Write combined dataframe to disk
    combined_frame = combined_frame.to_pickle("../data/cicids2017.pkl")

    # Clean up the root /data directory
    for index, path in enumerate(file_paths):
        os.remove(path)
    os.rmdir("../data/MachineLearningCSV/MachineLearningCVE/")
    os.rmdir("../data/MachineLearningCSV/")
    os.remove("../data/cicids2017.zip")
    os.remove("../data/MachineLearningCSV.md5")


# In[ ]:


combine_and_pickle()


# ### Method 2: Automated Retrieval from Kaggle

# #### Requirements
# 
# You need to have an account with [Kaggle](https://www.kaggle.com/). Once you have an account, navigate to Kaggle and then to the 'Account' tab of your user profile. Scroll down until you find the button 'Create New API Token'. Use this button to download an API token that will allow you to retrieve the dataset using the following script. On a machine running Windows 10, place the Kaggle API token at `C:\Users\<Windows-username>\.kaggle\kaggle.json`.

# #### Automated Retrieval, Combination, and Pickling

# In[ ]:


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
    get_ipython().system('kaggle datasets download cicdataset/cicids2017 -q')

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


# In[ ]:


retrieve_combine_and_pickle()


# ## File Information
# 
# The [CICIDS2017 dataset](https://www.kaggle.com/cicdataset/cicids2017) is a collection of simulated packet capture events labelled as either benign or attack events. This dataset was generated by the Canadian Institute for Cybersecurity and the University of New Brunswick. The authors set up separate attack and victim networks, each being a complete network topology consisting of routers, firewalls, switches, and with nineteen end-user machines between the two. The authors used a machine-learning model to simulate benign network traffic for five consecutive days, during which they also manually executed a series of common web attacks against the victim network from the attack network. All network traffic that occured on the victim network during this time period was captured and labeled, resulting in the given dataset. 
# 
# The following is an example of records taken from the dataset.

# In[2]:


frame = pandas.read_pickle('../data/cicids2017.pkl')
frame.head()


# ## Data Exploration

# Before building any models it's important that we understand the nature of our data. In this section we'll explore the dataset for nuances and note any properties we find that could potentially impact our ability to apply anomaly detection techniques.

# We'll first load a fresh instance of the dataset.

# In[2]:


frame = pandas.read_pickle('../data/cicids2017.pkl')


# Let's start our exploration by assessing the dimensions of the given dataset.

# In[3]:


row_count = len(frame.index)
col_count = len(frame.columns)

print("Row Count : " + str(row_count))
print("Column Count : " + str(col_count))


# There are nearly 3,000,000 packet capture events stored in this dataset across 79 tracked columns. This is a good size for a dataset that should be large enough to contribute to the development of useful models while remaining small enough to allow for many experiments in a short amount of time.

# It's important to understand the nature of the columns being tracked. We have two categorical columns, one being the target label and the other being the destination port number. The other 77 columns are all numerical, with some being discrete and others being continuous.

# In[4]:


for index, type in enumerate(frame.dtypes):
    print(frame.columns[index] + " : " + str(frame.dtypes[index]))


# The data type of the Destination Port column in particular needs additional consideration. Port numbers are indeed integers, but they are categorical labels rather than quantitative numerical values.

# Let us now consider the different target labels provided by the dataset.

# In[5]:


frame[' Label'].unique()


# In[6]:


len(frame[' Label'].unique())


# The target labels indicate whether a particular packet was benign or if it resulted from an attempted web attack, and also specify what kind of attack was performed. There are 15 unique labels, one of which indicates a benign packet and the other 14 representing web attacks.

# Next let's check our data for any null values.

# In[7]:


for index, sum in enumerate(frame.isnull().sum()):
    if sum > 0:
        print(frame.columns[index] + " : " + str(sum) + " null values")


# It appears that a small selection of records are missing values in one column. We'll want to consider how best to handle records with missing values.

# Another important characteristic of our dataset to understand is the division of our data into labels. How many malicious packets are there in comparison to benign packets?

# In[8]:


label_counts = {}

for i in range(0, len(frame)):
    label = frame.loc[i][78]
    if label not in label_counts:
        label_counts[label] = 1
    else:
        label_counts[label] = label_counts[label] + 1
    
    if i % 100000 == 0:
        print(str(i) + " records processed")

label_counts


# 80.3% of all packets captured in this dataset were benign, leaving 19.7% to be malicious. Of the malicious packets, the vast majority were generated by some form of denial-of-service type attack.

# ## Preprocessing

# In this section we'll perform any tasks required to prepare our dataset for input into our models.

# Let's start by loading in a fresh instance of the dataset.

# In[2]:


frame = pandas.read_pickle('../data/cicids2017.pkl')


# The first thing that I'd like to address is a number of unnecessary space characters present in some column names but not in others.

# In[3]:


for col in frame.columns:
    print(col)


# Let's trim all leading and trailing whitespace from the column names.

# In[4]:


for index, column in enumerate(frame.columns):
    frame.columns.values[index] = frame.columns.values[index].strip()


# In[5]:


for col in frame.columns:
    print(col)


# The next problem to consider is that destination port addresses should be treated as categorical labels rather than as numerical values. Inspecting the data type of the 'Destination Port' column reveals a type of int64.

# In[6]:


frame['Destination Port'].dtype


# Let's convert the data type of this column to a string. This will help us to ensure that our models consider this column's data categorically rather than numerically.

# In[7]:


frame['Destination Port'] = frame['Destination Port'].apply(str)


# In[8]:


frame['Destination Port'].dtype


# We discovered during the data exploration step that a number of records are missing values for the column 'Flow Bytes/s'. It isn't clear whether these values are mislabelled zeros or just a metric that failed to be captured for those particular observations. The safest option, given the low count of records affected, would be to simply drop these records. 

# In[9]:


frame = frame.dropna()


# In[10]:


print("Row Count : " + str(len(frame.index)))


# In[11]:


for index, sum in enumerate(frame.isnull().sum()):
    if sum > 0:
        print(frame.columns[index] + " : " + str(sum) + " null values")


# Our dataset has now been very slightly reduced, but contains no null values.

# In addition to removing any records with null values, I would also like to remove two particular columns from our dataset that have the potential to be problematic when building our models. The incredibly large values in the columns 'Flow Bytes/s' and 'Flow Packets/s' cause overflow errors when building many of our models, and so I will be removing these two columns from the dataset.

# In[12]:


frame = frame.drop(['Flow Bytes/s', 'Flow Packets/s'], axis=1)


# Our next preprocessing step will be to handle the separation of our labels from the rest of our dataset.

# In[13]:


labels = pandas.DataFrame(frame['Label'].copy(), columns = ['Label'])


# In[14]:


labels.head()


# In[15]:


frame = frame.drop('Label', axis=1)


# In[16]:


frame.head()


# In addition to separating the target labels from the rest of the dataset, I'd like to reduce the number of unique target labels from 15 to 2, representing either an attack packet or a benign packet. Our goal in this work is to apply anomaly detection techniques to an intrusion detection dataset, and as such we are only interested in whether a particular packet is benign or malicious.
# 
# Let's create a copy of the labels dataframe with this added simplification. We'll use the label 0 to represent a benign packet and 1 to represent a malicious packet.

# In[17]:


modified_labels_list = []

for label in labels.values:
    if label == 'BENIGN':
        modified_labels_list.append(0)
    else:
        modified_labels_list.append(1)
        
modified_labels = pandas.DataFrame(modified_labels_list, columns = ['Label'])
modified_labels.head()


# With this change we can now frame the problem as an imbalanced binary classification task. Let's write our preprocessed dataset to disk, along with both the original labels and the simplified labels.

# In[18]:


frame.to_pickle("../data/refined-cicids2017.pkl")
labels.to_pickle("../data/original-labels.pkl")
modified_labels.to_pickle("../data/simplified-labels.pkl")


# ## Experiments

# In this section we perform a series of experiments applying anomaly detection techniques to an intrusion detection task. Our goal in each experiment is to build a model capable of identifying which packets are benign and which are malicious. In every case we perform the following tasks:
# 
# - load the preprocessed dataset
# - split the dataset into training and test sets 
# - build a model using the training set
# - make predictions against the test set
# - write the predictions to disk
# 
# We also attempt to identify the assumptions that a given algorithm will make 

# ### Experiment 1: Mahalanobis Distance

# In our first experiment we employ a technique that uses a point's proximity to the distribution of points in the dataset as a mechanism for detecting anomalies. The Mahalanobis distance of a point represents its distance from the given distribution. Using a vector of sample means and a vector of sample standard deviations computed from the training set we generate a centerpoint for the distribution. We then use the covariance matrix of the dataset's features to control the shape of the distribution around the centerpoint when computing Mahalanobis distances for each point in the test set. We finally establish a threshold distance away from the distribution beyond which any point is labeled an anomaly. This technique makes the assumption that an anomalous point is further away from the distribution than normal points. This assumption is not necessarily true for every dataset.

# We'll begin by loading our samples and targets, dropping the categorical 'Destination Port' column.

# In[2]:


X = pandas.read_pickle('../data/refined-cicids2017.pkl').drop(['Destination Port'], axis=1)
Y = pandas.read_pickle('../data/simplified-labels.pkl')


# Next, we'll split our dataset into a training set and a test set. We'll use an 80/20 split across all techniques for consistency.

# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)


# Then we'll use the training set to compute the metrics required for computing Mahalanobis distances. The inverse covariance matrix, along with the centerpoint, are the model. This process can be considered the training phase.

# In[4]:


covariance = numpy.cov(X_train, rowvar=0)
inverse_covariance = numpy.linalg.pinv(covariance)
centerpoint = numpy.mean(X_train, axis=0)
centerpoint


# Next we compute the Mahalanobis distance between each point in the test set and the training set's distribution. This is the testing phase.

# In[5]:


distances = []

for index, record in X_test.iterrows():
    p1 = record
    p2 = centerpoint
    distance = (p1 - p2).T.dot(inverse_covariance).dot(p1 - p2)
    distances.append(distance)

distances = numpy.array(distances)
distances


# Before making predictions we need to define a cutoff threshold beyond which an event is considered anomalous. Here we define the cutoff using a Chi-square distribution. This distribution is produced by the Mahalanobis distances. We then generate a list of the packet capture event indices that have been identified as anomalous.

# In[6]:


cutoff = scipy.stats.chi2.ppf(0.80, X_test.shape[1])
prediction_indices = numpy.where(distances > cutoff)[0]
prediction_indices


# This prediction format won't interface with our evaluation functions. Out next stip is to convert this list of anomalous event indices into a list of predicted labels across all of the test set.

# In[7]:


predictions = []

for index in range(0, len(X_test)):
    if index in prediction_indices:
        predictions.append(1)
    else:
        predictions.append(0)
        
predictions


# Let's put this all together, performing a grid search of the hyperparameter state space. In this search we'll experiment with the cutoff threshold as the main hyperparameter.

# In[2]:


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


# ### Experiment 2: Isolation Forest

# In this experiment we employ a technique that takes advantage of the idea that anomalies are few in number and distant from normal data points, thus making them easier to isolate. The isolation forest algorithm repeatedly selects a feature at random and then selects a random value for that feature within the range of allowable values, dividing the data along that random value for the given random feature and thereby partitioning the data at that point. The repeated process of partitioning smaller and smaller regions of space can be considered from the perspective of a tree structure, with leaves representing the smaller partitions produced by each division. This process occurs in a hierarchical fashion, with partitions of space at the same distance from the root node of the isolation tree being divided before partitions at lower levels.
# 
# At some point in this process every point will exist within its own partition, and a length can be computed from the root node to the isolating leaf to determine how many partitioning steps were required to isolate a given point. Points that are successfully isolated in fewer steps are then considered anomalous based on the assumption that anomalies are inherently easier to isolate. This algorithm makes the assumption that an anomalous point will be easier to isolate using random hierarchical partitioning, which is not necessarily true for every dataset.

# We'll begin by loading our samples and targets.

# In[2]:


X = pandas.read_pickle('../data/refined-cicids2017.pkl')
Y = pandas.read_pickle('../data/simplified-labels.pkl')


# Next, we'll split our dataset into a training set and a test set. We'll use an 80/20 split across all techniques for consistency.

# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)


# Then we define and train our model against the training set.

# In[4]:


isoforest = IsolationForest(n_estimators=100, verbose=1, warm_start=False).fit(X_train)


# We can now use our model to make predictions against the test set.

# In[5]:


predictions = isoforest.predict(X_test)


# Let's put this all together, performing a grid search of the hyperparameter state space. In this search we'll experiment with the number of estimators as the main hyperparameter.

# In[2]:


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


# ### Experiment 3: Multiple Linear Regression

# In this experiment we employ a very traditional technique for classification tasks that uses a linear model as a mechanism for detecting anomalies. The linear regression algorithm begins generating a linear model of the data by choosing a line at random. It then iterates by making small adjustments to the model based on the proximity of the line to each of the points in the dataset, attempting to minimize a squared error cost function in order to achieve the best possible fit to the training data. New points can be plotted against the linear model to generate predictions about their labels, represented by the Y axis of the linear model. This technique assumes that a linear model can portray the relationship between sample features and their labels, which is not necessarily true for every dataset.

# We'll begin by loading our samples and targets.

# In[2]:


X = pandas.read_pickle('../data/refined-cicids2017.pkl')
Y = pandas.read_pickle('../data/simplified-labels.pkl')


# Next, we'll split our dataset into a training set and a test set. We'll use an 80/20 split across all techniques for consistency.

# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)


# Then we define and train our model against the training set.

# In[4]:


regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# We can now use our model to make predictions against the test set.

# In[5]:


predictions = regressor.predict(X_test)


# Next we can evaluate our predictions against our test labels.

# In[6]:


score = roc_auc_score(Y_test, predictions)


# Finally, we write this model's predictions to disk for further analysis.

# In[7]:


numpy.save('../data/multiple-linear-regression-predictions.npy', predictions)
numpy.save('../data/multiple-linear-regression-targets.npy', Y_test)
numpy.save('../data/multiple-linear-regression-score.npy', score)


# ### Experiment 4: Principal Component Analysis and K-Nearest Neighbors

# In this experiment we employ both a technique for dimensionality reduction and a technique for anomaly detection.
# 
# First we employ a technique for dimensionality reduction called Principal Component Analysis that, at a high level, uses the variance of each feature to determine which features are more important than others and to generate a lower dimensional view of the data consisting of linear transformations of features deemed to be the most import. This technique makes the assumption that features with greater variance are the most important.
# 
# Next we employ a technique that uses the labels of other points in proximity to a given point to determine whether it is anomalous. The K-Nearest Neighbors algorithm computes the Euclidean distances from a given point to all other points in a dataset to determine its nearest neighbors. The algorithm will then poll the class labels of the given point's k nearest neighbors, using the majority occurrence to predict the label of the new point. Dimensionality reduction is important when making use of this algorithm as a way to greatly reduce the temporal complexity of both training and making predictions. This technique makes the assumption that an anomalous point is close to other anomalies, which is not necessarily true for every dataset.

# We'll begin by loading our samples and targets.

# In[2]:


X = pandas.read_pickle('../data/refined-cicids2017.pkl')
Y = pandas.read_pickle('../data/simplified-labels.pkl')


# Next, we'll split our dataset into a training set and a test set. We'll use an 80/20 split across all techniques for consistency.

# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)


# In[4]:


pca = PCA()
pca.fit(X_train)


# In[5]:


cumsum = numpy.cumsum(pca.explained_variance_ratio_)
d = numpy.argmax(cumsum > 0.99) + 1
d


# In[6]:


pca = PCA(n_components = d)
X_train_reduced = pca.fit_transform(X_train)


# Then we define and train our model against the training set.

# In[12]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_reduced, Y_train.values.ravel())


# We can now use our model to make predictions against the test set.

# In[13]:


predictions = knn.predict(X_test_reduced)


# Let's put this all together, performing a grid search of the hyperparameter state space. In this search we'll experiment with k as the main hyperparameter.

# In[2]:


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


# Finally, we write this model's predictions to disk for further analysis.

# In[14]:


numpy.save('../data/pca-knn-predictions.pkl', predictions)


# ### Experiment 5: Principal Component Analysis and Local Outlier Factor

# In this experiment we employ both a technique for dimensionality reduction and a technique for anomaly detection.
# 
# First we employ Principal Component Analysis for dimensionality reduction, which was described previously in the Experiment 5 subsection. Next we employ a technique that uses the local density surrounding a point as a mechanism for detecting anomalies. The local outlier factor of a point is a measure of local density estimated using the distances from a given point to its k nearest neighbors. A lower local outlier factor will correspond to a point existing in a lower density region of the state space. Dimensionality reduction is important when making use of this algorithm as a way to greatly reduce the temporal complexity of both training and making predictions.  This technique makes the assumption that an anomalous point will exist in a lower density region of the state space.

# In[2]:


X = pandas.read_pickle('../data/refined-cicids2017.pkl')
Y = pandas.read_pickle('../data/simplified-labels.pkl')


# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=True)


# In[4]:


pca = PCA()
pca.fit(X_train)


# In[5]:


cumsum = numpy.cumsum(pca.explained_variance_ratio_)
d = numpy.argmax(cumsum > 0.99) + 1
d


# In[9]:


pca = PCA(n_components = d)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)


# Then we define and train our model against the training set.

# In[7]:


lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
lof.fit(X_train_reduced, Y_train.values.ravel())


# In[11]:


predictions = lof.predict(X_test_reduced)


# Let's put this all together, performing a grid search of the hyperparameter state space. In this search we'll experiment with k as the main hyperparameter.

# In[2]:


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
    
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
    lof.fit(X_train_reduced, Y_train.values.ravel())
    
    predictions = lof.predict(X_test_reduced)
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

numpy.save('../data/pca-lof-predictions.npy', best_predictions)
numpy.save('../data/pca-lof-targets.npy', Y_test)
numpy.save('../data/pca-lof-score.npy', best_roc_auc_score)


# ## Evaluation

# In this section we evaluate the performance of each anomaly detection technique.

# We'll start by loading all of our model predictions for further evaluation.

# In[2]:


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


# Next we'll force all predictions to encode as 0 or 1.

# In[3]:


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


# The first metric I'd like to compute is the percentage of anomalies accurately labeled by each technique out of all anomalies. This metric is known as recall and will help us to immediately understand how well our models were able to detect anomalies.

# In[4]:


mahalanobis_recall = recall_score(mahalanobis_targets, mahalanobis_predictions)
isoforest_recall = recall_score(isoforest_targets, isoforest_predictions)
regression_recall = recall_score(regression_targets, regression_predictions)
pca_knn_recall = recall_score(pca_knn_targets, pca_knn_predictions)
pca_lof_recall = recall_score(pca_lof_targets, pca_lof_predictions)


# Another important metric for us to compute is precision. This will show us how many of the instances that we flagged as anomalies were actually anomalies.

# In[5]:


mahalanobis_precision = precision_score(mahalanobis_targets, mahalanobis_predictions)
isoforest_precision = precision_score(isoforest_targets, isoforest_predictions)
regression_precision = precision_score(regression_targets, regression_predictions)
pca_knn_precision = precision_score(pca_knn_targets, pca_knn_predictions)
pca_lof_precision = precision_score(pca_lof_targets, pca_lof_predictions)


# Our final metric will embody both precision and recall in a single score. The receiver operating characteristic curve is a plot of recall vs. precision at all thresholds. The AUROC metric takes the area under the ROC curve, presenting a simple metric between 0 and 1.0 for evaluating classification performance.

# In[6]:


mahalanobis_auroc = roc_auc_score(mahalanobis_targets, mahalanobis_predictions)
isoforest_auroc = roc_auc_score(isoforest_targets, isoforest_predictions)
regression_auroc = roc_auc_score(regression_targets, regression_predictions)
pca_knn_auroc = roc_auc_score(pca_knn_targets, pca_knn_predictions)
pca_lof_auroc = roc_auc_score(pca_lof_targets, pca_lof_predictions)


# Let's combine each of these metrics into a single table.

# In[7]:


name_series    = pandas.Series(['Mahalanobis Distances', 'Isolation Forests', 'Multiple Linear Regression', 'PCA K-Nearest Neighbors', 'PCA Local Outlier Factor'])
precision_series  = pandas.Series([mahalanobis_precision, isoforest_precision, regression_precision, pca_knn_precision, pca_lof_precision ])
recall_series  = pandas.Series([mahalanobis_recall, isoforest_recall, regression_recall, pca_knn_recall, pca_lof_recall ])
auroc_series = pandas.Series([mahalanobis_auroc, isoforest_auroc, regression_auroc, pca_knn_auroc, pca_lof_auroc ])

metrics_frame = pandas.DataFrame({'Method' : name_series, 'Precision' : precision_series, 'Recall' : recall_series, 'AUROC' : auroc_series })
metrics_frame


# For the evaluation of our models we have collected a series of three metrics, being precision, recall, and AUROC. As it relates to our intrusion detection task specifically, the recall metric is the most important to maximize. The primary goal of our task is to detect as many of the anomalies as possible to prevent attacks from causing damage to a system. With this in mind, it is made clear by our performance metrics that the Isolation Forest algorithm was the best at detecting anomalies, boasting a recall of 90.9%. The Local Outlier Factor method comes in second at a respectable 81.0% while the other three methods come up short.
# 
# The methods with the lowest recall, being the Mahalanobis Distance method and the K-Nearest Neighbors method, tell us something about the nature of anomalous points in our dataset. These scores imply that malicous packet events occur in relatively close proximity to the distribution of packet events, and that the nearest neighbors to a malicious event plotted in our dataset's state space are often benign events rather than anomalies. This is an interesting characteristic in that the anomalies are apparently able to blend in, to some extent, with normal events.
# 
# On the contrary, our two highest recall methods can also shed some light on the nature of our anomalous data. The Local Outlier Factor method relies directly on low local area densities to detect anomalies while the Isolation Forest method indirectly uses local area density to isolate anomalies in fewer steps than normal data. This implies that even if our network intrusion events occur in close proximity to benign events, they often tend to occur in areas of low density. This is a useful characteristic of malicious events to extract from these metrics.
# 
# Taking a look at the precision column of our metric table reveals that most of our models produced a problematic number of false alarms. Even the Isolation Forest model with its impressive recall suffers from a precision of only 19.7%. This indicates that our best models have room to grow with additional refinement and tuning.
# 
# It's very interesting to note the one model that attained a high level of precision along with a recall that isn't as abysmal as other models. The Multiple Linear Regression model achieved a precision of 91.0% and a recall of 47.4%. Looking at the AUROC metrics for each of our models, we can see that the Multiple Linear Regression model comes away with the best combined performance. This is promising. The intuitive next step beyond linear regression is the artificial neural network, and with these results my next step would be to try training a neural network for this task.

# ## Conclusions

# Machine learning techniques for anomaly detection can perform well on intrusion detection tasks. An analysis of our model performance indicates that attacks on our victim network are located in close proximity to benign network traffic events within the state space of our dataset, but that they tend to lie in regions of low density and that his characteristic can be exploited by certain types of models to detect anomalies with a high rate of recall. We also discovered that most of our models trigger an unsustainable number of false alarms and would require additional refinement.
# 
# Further work in this area should not only continue to refine the existing models defined here, but would also attempt to apply new types of models to the same task. One such model that seems likely to be applied successfully is the artificial neural network. Another interesting type of model that might be fruitful when applied to our specific task would be an angle-based model. This model would compute the angles between a new point and pairs of its k nearest neighbors, using the assumption that points with smaller angles measured in such a manner are more likely to be anomalies. This may work well with the given dataset due to the tendency for anomalous points to exist in low density regions of the state space.

# ## References
# 
# [1] Cansiz, S. (2021, April 17). Multivariate Outlier Detection in Python. Medium. https://towardsdatascience.com/multivariate-outlier-detection-in-python-e946cfc843b3. 
# 
# [2] Harris, C. R., Millman, K. J., Walt, S. J. van der, Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020, September 16). Array programming with NumPy. Nature News. https://www.nature.com/articles/s41586-020-2649-2. 
# 
# [3] Pasricha, S. (2020, November). Anomaly Detection and Security. Embedded Systems and Machine Learning. Fort Collins, CO; Colorado State University. 
# 
# [4] Pedregosa, F., Profile, V., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., … Authors:   Fabian Pedregosa  View Profile. (2011, November 1). Scikit-learn: Machine Learning in Python. The Journal of Machine Learning Research. https://dl.acm.org/doi/10.5555/1953048.2078195. 
# 
# [5] Sharafaldin, I., Habibi Lashkari, A., &amp; Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. Proceedings of the 4th International Conference on Information Systems Security and Privacy. https://doi.org/10.5220/0006639801080116 
# 
# [6] Virtanen P;Gommers R;Oliphant TE;Haberland M;Reddy T;Cournapeau D;Burovski E;Peterson P;Weckesser W;Bright J;van der Walt SJ;Brett M;Wilson J;Millman KJ;Mayorov N;Nelson ARJ;Jones E;Kern R;Larson E;Carey CJ;Polat İ;Feng Y;Moore EW;VanderPlas J;Laxalde D;P. (n.d.). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods. https://pubmed.ncbi.nlm.nih.gov/32015543/. 

# ## About this notebook
# 
# **Author:** Christian Westbrook, Colorado State University  
# **Published:** April 28th, 2020
