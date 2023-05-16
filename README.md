# Phishing Website Detection



Table of Contents
Overview
Installation
Usage
File Descriptions

Overview
With the rise of electronic trading, unauthorized users are increasingly accessing sensitive information leading to financial losses for individuals and corporations alike. This project aims to address the problem of phishing websites by using machine learning techniques for identifying such attempts accurately and effectively.

Installation
To clone and run this application, you'll need Git and Python (which comes with pip) installed on your computer. From your command line

This project uses the following python libraries:
pandas
numpy
sklearn
joblib
warnings
matplotlib
prettytable

Run featuremodel.py to identify the number of features required by PCA, KBest, and RFECV. This process can be time-consuming due to the extensive computations required.
# Run the featuremodel.py
$ python featuremodel.py

#As the Model Takes Long time to run I have integrated the Values received from this script in to the next file Directly

Use the values obtained from featuremodel.py in model_training.py to train the models and save the joblib files.
#If We want to use the Values Directly generated from the Featuremodel.py uncomment #from feature_selection import best_k_selectkbest,best_k_rfe,best_n_pca from the model_training File and Comment these Variables best_k_selectkbest = 49,
best_n_pca = 53,best_k_rfe = 55.

# Run the model_training.py
$ python model_training.py

Run evaluation.py to evaluate the trained models and visualize the results and performance metrics.
# Run the evaluation.py
$ python evaluation.py

File Descriptions
featuremodel.py: This script identifies the optimal number of features required by PCA, KBest, and RFECV.
model_training.py: This script utilizes the output of featuremodel.py to train machine learning models and save them as joblib files.
evaluation.py: This script loads the trained models and evaluates them, providing both statistical results and graphical visualizations.

