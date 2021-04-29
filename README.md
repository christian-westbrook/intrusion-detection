# Applications of Anomaly Detection Techniques to Intrusion Detection

*A data science project by Christian Westbrook.*

Intrusion detection is a common cybersecurity task involving the inspection of a networked environment for evidence of malicious activity. One approach to intrusion detection is to view the problem as a need to identify anomalous network activity. In this work we demonstrate that idea by performing a series of experiments applying known machine learning techniques for anomaly detection to an intrusion detection task. We employ the CICIDS2017 dataset of labeled network intrusion events in the training of our models. We set out to compare the effectiveness of multiple anomaly detection techniques in the task of identifying malicious packet events from the given dataset. To this end we experiment with applications of Mahalanobis distances, K-Nearest Neighbors, Local Outlier Factor, Isolation Forests, Multiple Linear Regression, and Principal Component Analysis to the network intrusion task. We found that models operating under the assumption that anomalies exist at a distance from normal data points scored poorly in recall while models operating under the assumption that anomalies exist in low density regions of the state space scored well in recall. We also discovered that most of our models struggled with precision, triggering an excessively high number of false alarms.

## Requirements

Your machine needs to have a recent version of Python and a recent version of pip installed to use the source code in this repository.

Python: https://www.python.org/
pip: https://pypi.org/project/pip/

If you plan on using the automated dataset retrieval feature, either in the notebook or with the retrieve.py script, you need to do the following:

- Create an account at https://www.kaggle.com/
- Navigate to the 'Account' tab of your user profile
- Scroll down until you find the button 'Create New API Token'
- Use this button to download an API token
- If you're using Windows 10 place the token at `C:\Users\<Windows-username>\.kaggle\kaggle.json`
- If you're not using Windows, refer to Kaggle's documentation to find where to place your token

The token will need to be in place for the automated retrieval process to work. These steps are repeated in the notebook and in the installation section of this README.

This project was designed for Windows 10 machines. However, almost all of the source code here is compatible with both MacOS and Linux. If you want to work with this repository on either MacOS or Linux you only need to convert the script install.bat into a script suitable for your operating system.

Our jupyter notebook requires an operating system with a GUI to view. Our scripts will work on any operating system with Python support.

## Overview of Project Assets

This section contains a breakdown of the assets available for use and review in this repository, along with instructions about how to configure your environment to make use of each asset.

### 1. Jupyter Notebook

A jupyter notebook is made available in the `notebook` directory. To prepare your machine for using this notebook you will need to install its dependencies. You can do this in a few ways.  

There is a Windows batch script in the root `scripts` directory called install.bat. This script will first confirm that you have Python and pip installed and then attempt to install all dependencies. If you run this script before running the notebook you'll have everything you need.

If you're comfortable with Python and pip, you can simply install the following dependencies yourself:

- notebook
- kaggle
- pandas
- numpy
- sklearn
- scipy

Once you have all dependencies installed, use the command 'jupyter notebook' in your terminal to launch jupyter notebook. For more information about how to work with jupyter notebooks refer to https://jupyter.org/

### 2. Scripts

A set of scripts are made available in the root `script` directory. What follows is a list of script names and their descriptions in operational order.

NOTE: The data required for the evaluation script comes with the repository. You need to retrieve and preprocess the dataset to use the experiment functions, but the outcome of each experiment is made available out of the box and can be used in the evaluate script. Running these experiments yourself will produce new performance metrics due to the stochastic nature of training a model with a shuffled training set and test set.

install.bat
- Checks for your Python and pip versions
- Installs all other project dependencies
- Requires Windows

retrieve.py
- Requires that your Kaggle token is in place (refer to the installation section below)
- Automatically retrieves the dataset and places it in the root `/data` directory

preprocess.py
- Requires that the dataset exists in the root `/data` directory
- Prepares the dataset for use in the experiment scripts

ex1_mahalanobis.py
- Requires that the preprocessed dataset exist in the root `/data` directory
- Performs the Mahalanobis distance experiment
- Outputs the results into the root `/data` directory

ex2_isoforest.py
- Requires that the preprocessed dataset exist in the root `/data` directory
- Performs the Isolation Forest experiment
- Outputs the results into the root `/data` directory

ex3_regression.py
- Requires that the preprocessed dataset exist in the root `/data` directory
- Performs the Multiple Linear Regression experiment
- Outputs the results into the root `/data` directory

ex4_pca_knn.py
- Requires that the preprocessed dataset exist in the root `/data` directory
- Performs the K-Nearest Neighbors experiment
- Outputs the results into the root `/data` directory

ex5_pca_lof.py
- Requires that the preprocessed dataset exist in the root `/data` directory
- Performs the Local Outlier Factor experiment
- Outputs the results into the root `/data` directory

evaluate.py
-  Requires that the output of all five experiments exists in the root `/data` directory
-- Note that this data comes with the repository
- Prints out precision, recall, and AUROC for all experiments

### 3. Report

A report is made available in the `report` directory. The .pdf file is the final report. All other files in that directory relate to formatting with LaTeX.

### 4. Repository

A GitHub repository for this project is located at https://github.com/christian-westbrook/intrusion-detection

## Installation

To prepare your machine for using the source code in this repository, you will need to ensure both that you have already installed a recent version of Python and a recent version of pip.

Python: https://www.python.org/  
pip: https://pypi.org/project/pip/

Your machine needs to have a recent version of Python and a recent version of pip installed to use the source code in this repository.

Python: https://www.python.org/
pip: https://pypi.org/project/pip/

If you plan on using the automated dataset retrieval feature, either in the notebook or with the retrieve.py script, you need to do the following:

- Create an account at https://www.kaggle.com/
- Navigate to the 'Account' tab of your user profile
- Scroll down until you find the button 'Create New API Token'
- Use this button to download an API token
- If you're using Windows 10 place the token at `C:\Users\<Windows-username>\.kaggle\kaggle.json`
- If you're not using Windows, refer to Kaggle's documentation to find where to place your token

The token will need to be in place for the automated retrieval process to work.

Once you have Python and pip installed and your token in place it's time to install the rest of the dependencies. You can do this in a few ways.  

There is a Windows batch script in the root `scripts` directory called install.bat. This script will first confirm that you have Python and pip installed and then attempt to install all dependencies. If you run this script before running the notebook you'll have everything you need.

If you're comfortable with Python and pip, you can simply install the following dependencies yourself:

- notebook
- kaggle
- pandas
- numpy
- sklearn
- scipy

Once you have all dependencies installed, use the command 'jupyter notebook' in your terminal to launch jupyter notebook. For more information about how to work with jupyter notebooks refer to https://jupyter.org/

Or, if you'd rather use our scripts, refer to the scripts section of the assets overview above.
