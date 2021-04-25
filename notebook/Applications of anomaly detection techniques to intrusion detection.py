#!/usr/bin/env python
# coding: utf-8

# # Applications of Anomaly Detection Techniques to Intrusion Detection

# ## Before You Begin
# 
# This section contains important information about executing the code in this notebook along with an alternative way to access the source code. If you don't plan on executing any code yourself, feel free to skip this section.
# 
# Some of the source code in this notebook makes assumptions about where you are running it. If you are planning on executing the code in this notebook, ensure that you are either running it within a clone of the [associated repository](https://github.com/christian-westbrook/intrusion-detection) at its default location, or that you understand how to adapt relative file paths to meet your needs. A good way to determine if a block of code makes this assumption is to look for instances where the `os` module is being used. If you aren't sure, feel free to clone a fresh copy of the repository and run your new copy of the notebook at its default location.
# 
# This notebook contains more than just source code. If you're only interested in using the source code, you may prefer to use the application instead of this notebook. The application is located in the root `/application` directory and presents a streamlined and interactive experience in comparison to this notebook. For instructions on how to use the application, refer to the README located in the root directory of the repository.

# ## Imports
# 
# - *pandas* for processing and rendering tabular data
# - *glob* for enabling dynamic interfacing with the file system
# - *kaggle* for interfacing with [Kaggle](https://www.kaggle.com/)
# - *os* for interfacing with the machine where this notebook is being ran
# - *zipfile* for managing .zip archives
# - *tensorflow* for tensor processing
# - *keras* for simplified tensor processing

# In[1]:


# Data Management
import pandas

# External Interfaces
import glob
import kaggle
import os
from zipfile import ZipFile

# TensorFlow
import tensorflow
from tensorflow import keras


# ## Introduction
# 
# The goal of this work is to explore how existing anomaly detection techniques can be applied to the domain of intrusion detection. To this end we perform a series of experiments testing known anomaly detection techniques against the CICIDS2017 dataset of network intrusion events. We begin by exploring the given dataset and preprocessing in order to prepare the dataset for input into our models.

# ## Defining Terms

# ***CICIDS2017*** - A dataset of simulated packet capture events containing both benign and attack events.

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

# In[3]:


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


# In[4]:


combine_and_pickle()


# ### Method 2: Automated Retrieval from Kaggle

# #### Requirements
# 
# You need to have an account with [Kaggle](https://www.kaggle.com/). Once you have an account, navigate to Kaggle and then to the 'Account' tab of your user profile. Scroll down until you find the button 'Create New API Token'. Use this button to download an API token that will allow you to retrieve the dataset using the following script. On a machine running Windows 10, place the Kaggle API token at `C:\Users\<Windows-username>\.kaggle\kaggle.json`.

# #### Automated Retrieval, Combination, and Pickling

# In[2]:


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


# In[3]:


retrieve_combine_and_pickle()


# ## File Information
# 
# The [CICIDS2017 dataset](https://www.kaggle.com/cicdataset/cicids2017) is a collection of simulated packet capture events labelled as either benign or attack events. This dataset was generated by the Canadian Institute for Cybersecurity and the University of New Brunswick. The authors set up separate attack and victim networks, each being a complete network topology consisting of routers, firewalls, switches, and roughly one dozen end-user machines. The authors used a machine-learning model to simulate benign network traffic for five consecutive days, during which they also manually executed a series of common web attacks against the victim network from the attack network. All network traffic that occured on the victim network during this time period was captured and labelled, resulting in the given dataset. 
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

# Our next preprocessing step will be to handle the separation of our labels from the rest of our dataset.

# In[12]:


labels = pandas.DataFrame(frame['Label'].copy(), columns = ['Label'])


# In[13]:


labels.head()


# In[14]:


frame = frame.drop('Label', axis=1)


# In[15]:


frame.head()


# In addition to separating the target labels from the rest of the dataset, I'd like to reduce the number of unique target labels from 15 to 2, representing either an attack packet or a benign packet. Our goal in this work is to apply anomaly detection techniques to an intrusion detection dataset, and as such we are only interested in whether a particular packet is benign or malicious.
# 
# Let's create a copy of the labels dataframe with this added simplification. We'll use the label 0 to represent a benign packet and 1 to represent a malicious packet.

# In[16]:


modified_labels_list = []

for label in labels.values:
    if label == 'BENIGN':
        modified_labels_list.append(0)
    else:
        modified_labels_list.append(1)
        
modified_labels = pandas.DataFrame(modified_labels_list, columns = ['Label'])
modified_labels.head()


# With this change we can now frame the problem as an imbalanced binary classification task. Let's write our preprocessed dataset to disk, along with both the original labels and the simplified labels.

# In[17]:


frame.to_pickle("../data/refined-cicids2017.pkl")
labels.to_pickle("../data/original-labels.pkl")
modified_labels.to_pickle("../data/simplified-labels.pkl")


# ## Statistical Methods

# In this section we will focus on applications of statistical methods for anomaly detection to our intrusion detection task.

# ### Mahalanobis Distance

# In[ ]:




