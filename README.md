# Enhanced Phishing Website Detection System

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation Instructions](#installation-instructions)
3. [Application Usage](#application-usage)
4. [File Descriptions](#file-descriptions)

<a name="project-overview"></a>
## 1. Project Overview
In the contemporary era of e-commerce, the unauthorized intrusion into sensitive data is rapidly increasing, resulting in significant financial implications for individuals and businesses alike. This project is designed to tackle the persistent problem of phishing websites, deploying sophisticated machine learning methodologies to accurately and efficiently detect such malicious attempts.

Data Source: https://data.mendeley.com/datasets/c2gw7fy2j4/3

<a name="installation-instructions"></a>
## 2. Installation Instructions
To clone and operate this application, ensure that you have Git and Python, which includes pip, installed on your device. Use your command line for this process.

# install required Dependencies
$ pip install -r requirements.txt

Execute `featuremodel.py` to determine the number of features needed by PCA, KBest, and RFECV. Please note that this process can be quite time-intensive due to the extensive computations involved.

# Run featuremodel.py
$ python featuremodel.py

Considering the extended runtime of the Model, the values derived from this script have been directly incorporated into the subsequent file.

<a name="application-usage"></a>
## 3. Application Usage
Utilize the values derived from `featuremodel.py` in `model_training.py` to train the models and save the joblib files.

If you wish to use the values directly generated from `featuremodel.py`, uncomment `#from feature_selection import best_k_selectkbest, best_k_rfe, best_n_pca` from the `model_training.py` file and comment these variables `best_k_selectkbest = 49`, `best_n_pca = 53`, `best_k_rfe = 55`.

# Run model_training.py
$ python model_training.py


Run `evaluation.py` to evaluate the performance of the trained models and visualize the results and performance metrics.

# Run evaluation.py
$ python evaluation.py


<a name="file-descriptions"></a>
## 4. File Descriptions
- `featuremodel.py`: This script identifies the optimal number of features required by PCA, KBest, and RFECV.
- `model_training.py`: This script uses the output from `featuremodel.py` to train the machine learning models and saves them as joblib files.
- `evaluation.py`: This script loads the trained models and evaluates their performance, providing comprehensive statistical results and graphical visualizations.


# Team Contribution:
| Task                 | Team Members               |
|----------------------|----------------------------|
| Project Understanding| Megha, Venkatasai, Sankeerth|
| Data Understanding   | Venkatasai, Sankeerth      |
| Data Collection      | Megha                      |
| Data Preparation     | Megha, Sankeerth           |
| Data Analysis        | Venkatsai
| Modelling            | Megha, Venkatasai, Sankeerth|
| Evaluation & Results | Sankeerth                  |
| Documentation        | Megha, Venkatasai          |



