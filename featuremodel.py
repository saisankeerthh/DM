import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

# Load data
df = pd.read_csv("/content/dataset_full.csv")

# Clean data (if necessary)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Split data into train and test sets
X = df.drop(['phishing'], axis=1)
y = df['phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Remove constant features using VarianceThreshold
selector = VarianceThreshold()
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

# RFE with cross-validated selection
rfecv = RFECV(estimator=RandomForestClassifier(random_state=1), scoring='f1_macro', cv=5)
X_train_rfecv = rfecv.fit_transform(X_train, y_train)

# Print the optimal number of features and the corresponding score
print(f"Optimal number of features for RFECV: {rfecv.n_features_}, with F1-macro score: {rfecv.cv_results_['mean_test_score'].max():.4f}")

# Find the optimal number of features for SelectKBest using binary search
def binary_search_kbest(X_train, y_train, lower, upper):
    if lower == upper:
        return lower

    mid = (lower + upper) // 2
    kbest_low = SelectKBest(mutual_info_classif, k=mid)
    X_train_low = kbest_low.fit_transform(X_train, y_train)
    rfc_low = RandomForestClassifier(random_state=1)
    scores_low = cross_val_score(rfc_low, X_train_low, y_train, cv=5, scoring='f1_macro',n_jobs=-1)
    mean_score_low = np.mean(scores_low)

    kbest_high = SelectKBest(mutual_info_classif, k=mid + 1)
    X_train_high = kbest_high.fit_transform(X_train, y_train)
    rfc_high = RandomForestClassifier(random_state=1)
    scores_high = cross_val_score(rfc_high, X_train_high, y_train, cv=5, scoring='f1_macro',n_jobs=-1)
    mean_score_high = np.mean(scores_high)

    if mean_score_low < mean_score_high:
        return binary_search_kbest(X_train, y_train, mid + 1, upper)
    else:
        return binary_search_kbest(X_train, y_train, lower, mid)

optimal_k_kbest = binary_search_kbest(X_train, y_train, 1, X_train.shape[1])
print(f"Optimal number of features for SelectKBest: {optimal_k_kbest}")

# Find the optimal number of components for PCA using binary search
def binary_search_pca(X_train, y_train, lower, upper):
    if lower == upper:
        return lower

    mid = (lower + upper) // 2
    pca_low = PCA(n_components=mid)
    X_train_low = pca_low.fit_transform(X_train)
    rfc_low = RandomForestClassifier(random_state=1)
    scores_low = cross_val_score(rfc_low, X_train_low, y_train, cv=5, scoring='f1_macro',n_jobs=-1)
    mean_score_low = np.mean(scores_low)
    
    pca_high = PCA(n_components=mid + 1)
    X_train_high = pca_high.fit_transform(X_train)
    rfc_high = RandomForestClassifier(random_state=1)
    scores_high = cross_val_score(rfc_high, X_train_high, y_train, cv=5, scoring='f1_macro',n_jobs=-1)
    mean_score_high = np.mean(scores_high)

    if mean_score_low < mean_score_high:
        return binary_search_pca(X_train, y_train, mid + 1, upper)
    else:
        return binary_search_pca(X_train, y_train, lower, mid)

optimal_k_pca = binary_search_pca(X_train, y_train, 1, X_train.shape[1])
print(f"Optimal number of components for PCA: {optimal_k_pca}")