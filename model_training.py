# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
#from feature_selection import best_k_selectkbest,best_k_rfe,best_n_pca
# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/saisankeerthh/DM/main/dataset_full.csv")
df_original = df.copy()  # Make a copy of original dataframe for later use

# Clean the data by replacing infinity values and NaNs with zero
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Generate correlation matrix to understand relationships between features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

# Drop highly correlated features to avoid multicollinearity
high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
df.drop(high_corr, axis=1, inplace=True)

# Apply Variance Threshold for feature selection (remove low variance features)
vt = VarianceThreshold(threshold=0.1)
vt.fit(df)

# Select the features which passed the variance threshold
cols = df.columns[vt.get_support(indices=True)]
df = df[cols]

# Split the cleaned data into train and test sets
X = df.drop(['phishing'], axis=1)
y = df['phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Scale the features to have similar range of values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Also split original data (before cleaning) into train and test sets
X_original = df_original.drop(['phishing'], axis=1)
y_original = df_original['phishing']
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=1)

# Set parameters for feature selection methods
best_k_selectkbest = 49
best_n_pca = 53
best_k_rfe = 55

# Apply SelectKBest feature selection
k_best = SelectKBest(mutual_info_classif, k=best_k_selectkbest)
X_train_kbest = k_best.fit_transform(X_train_original, y_train_original)
X_test_kbest = k_best.transform(X_test_original)

# Apply PCA for feature extraction
pca = PCA(n_components=best_n_pca)
X_train_pca = pca.fit_transform(X_train_original)
X_test_pca = pca.transform(X_test_original)

# Apply Recursive Feature Elimination (RFE) for feature selection
rfc = RandomForestClassifier(random_state=1)
rfecv = RFECV(estimator=rfc, min_features_to_select=best_k_rfe)
X_train_rfe = rfecv.fit_transform(X_train_original, y_train_original)
X_test_rfe = rfecv.transform(X_test_original)

# Save all processed data to csv files for future use
pd.DataFrame(X_test).to_csv("X_test_all.csv", index=False)
pd.DataFrame(X_train).to_csv("X_train_all.csv", index=False)
pd.DataFrame(X_test_kbest).to_csv("X_test_kbest.csv", index=False)
pd.DataFrame(X_train_kbest).to_csv("X_train_kbest.csv", index=False)
pd.DataFrame(X_test_pca).to_csv("X_test_pca.csv", index=False)
pd.DataFrame(X_train_pca).to_csv("X_train_pca.csv", index=False)
pd.DataFrame(X_test_rfe).to_csv("X_test_rfe.csv", index=False)
pd.DataFrame(X_train_rfe).to_csv("X_train_rfe.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

# Define a list of models to be trained with their respective data
models = [('Decision Tree', DecisionTreeClassifier(), X_train, X_test),
          ('Decision Tree kbest', DecisionTreeClassifier(), X_train_kbest, X_test_kbest),
          ('Decision Tree RFE', DecisionTreeClassifier(), X_train_rfe, X_test_rfe),
          ('Decision Tree PCA', DecisionTreeClassifier(), X_train_pca, X_test_pca),
          ('Random Forest', RandomForestClassifier(), X_train, X_test),
          ('Random Forest kbest', RandomForestClassifier(), X_train_kbest, X_test_kbest),
          ('Random Forest RFE', RandomForestClassifier(), X_train_rfe, X_test_rfe),
          ('Random Forest PCA', RandomForestClassifier(), X_train_pca, X_test_pca),
          ('Naive Bayes', GaussianNB(), X_train, X_test),
          ('Naive Bayes kbest', GaussianNB(), X_train_kbest, X_test_kbest),
          ('Naive Bayes RFE', GaussianNB(), X_train_rfe, X_test_rfe),
          ('Naive Bayes PCA', GaussianNB(), X_train_pca, X_test_pca),
          ('AdaBoost ', AdaBoostClassifier(), X_train, X_test),
          ('AdaBoost kbest', AdaBoostClassifier(), X_train_kbest, X_test_kbest),
          ('AdaBoost RFE', AdaBoostClassifier(), X_train_rfe, X_test_rfe),
          ('AdaBoost PCA', AdaBoostClassifier(), X_train_pca, X_test_pca)]

trained_models = []

# Train all the models and save them in the list of trained_models
for name, model, X_train, X_test in models:
    model.fit(X_train, y_train)
    trained_models.append((name, model))

# Save all trained models to disk for future use
for name, model in trained_models:
    joblib.dump(model, f"{name}.joblib")

