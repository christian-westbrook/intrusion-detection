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
# - *kaggle* for interfacing with [Kaggle](https://www.kaggle.com/)
# - *os* for interfacing with the machine where this notebook is being ran
# - *zipfile* for managing .zip archives
# - *tensorflow* for tensor processing
# - *keras* for simplified tensor processing

# In[1]:


# Data Management
import pandas

# External Interfaces
import kaggle
import os
from zipfile import ZipFile

# TensorFlow
import tensorflow
from tensorflow import keras


# ## Introduction
# 
# The goal of this work is to explore how existing anomaly detection techniques can be applied to the domain of intrusion detection and to perform a comparison of techniques when evaluated against a dataset of intrusion events.

# ## Defining Terms

# ***CICIDS2017*** - A dataset of simulated packet capture events containing both benign and attack events.

# ## Loading Data

# There's more than one way to retreive the dataset. Use any of the following methods to retrieve the dataset and prepare it for use in the rest of the notebook.

# ### Manual Retrieval from Kaggle

# The dataset is retrievable from [a hosted location on Kaggle](https://www.kaggle.com/cicdataset/cicids2017). To prepare the data for use in this notebook, start by navigating to the dataset on Kaggle using the following link and downloading the dataset. This step will require an account with Kaggle.
# 
# https://www.kaggle.com/cicdataset/cicids2017
# 
# Extract the contents of this .zip archive file into the root `/data` directory. Explore the contents of the extracted archive and move all .csv files from their sub-directories into the root `/data` directory. This notebook will expect that all .csv files from the dataset are located in the root `/data` directory.

# ### Automated Retrieval from Kaggle

# Follow these steps to automate the retrieval of the dataset from Kaggle.

# #### Requirements
# 
# You need to have an account with [Kaggle](https://www.kaggle.com/). Once you have an account, navigate to Kaggle and then to the 'Account' tab of your user profile. Scroll down until you find the button 'Create New API Token'. Use this button to download an API token that will allow you to retrieve the dataset using the following script. On a machine running Windows 10, place the Kaggle API token at `C:\Users\<Windows-username>\.kaggle\kaggle.json`.

# #### Automated Retrieval

# In[ ]:


# Define all of the CSV files expected from the dataset
files = [
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv'
]

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

# Move all CSV files from the unzipped folder structure into the root /data directory
for index, file in enumerate(files):
    os.replace("../data/MachineLearningCSV/MachineLearningCVE/" + file, "../data/" + file)
    
# Read each CSV into a pandas dataframe
frames = []
for index, file in enumerate(files):
    frames.append(pandas.read_csv("../data/" + file))

# Merge dataframes vertically
combined_frame = pandas.concat(frames, axis=0)

# Write combined dataframe to disk
combined_frame.to_pickle("../data/cicids2017.pkl")

# Clean up the root /data directory
for index, file in enumerate(files):
    os.remove("../data/" + file)
os.rmdir("../data/MachineLearningCSV/MachineLearningCVE/")
os.rmdir("../data/MachineLearningCSV/")
os.remove("../data/cicids2017.zip")
os.remove("../data/MachineLearningCSV.md5")


# ## File Information
# 
# The CICIDS2017 dataset is a collection of simulated packet capture events labelled as either benign or attack events. The following is an example of records taken from the dataset.

# In[ ]:


frame = pandas.read_pickle('../data/cicids2017.pkl')
frame.head()


# ## Data Exploration

# ## Preprocessing
