# Suppress any warnings to maintain clean output
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, recall_score, precision_score
import joblib
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Load trained models from file
model_names = ["Decision Tree", "Decision Tree kbest", "Decision Tree RFE", "Decision Tree PCA", 
               "Random Forest", "Random Forest kbest", "Random Forest RFE", "Random Forest PCA", 
               "Naive Bayes", "Naive Bayes kbest", "Naive Bayes RFE", "Naive Bayes PCA", 
                "Adaboost kbest", "Adaboost RFE", "Adaboost PCA"]

trained_models = [(name, joblib.load(f"{name}.joblib")) for name in model_names]

# Load preprocessed test data from file
X_test_all = pd.read_csv("X_test_all.csv")
X_test_kbest = pd.read_csv("X_test_kbest.csv")
X_test_rfe = pd.read_csv("X_test_rfe.csv")
X_test_pca = pd.read_csv("X_test_pca.csv")
y_test = pd.Series(pd.read_csv("y_test.csv").iloc[:, 0])

# Initialize a table to store the evaluation results
results = PrettyTable(["Model", "Accuracy", "AUC", "F1 Score", "Recall", "Precision"])

# Evaluate the performance of each model on the test set and record the results
for name, model in trained_models:
    if "kbest" in name:
        X_test = X_test_kbest
    elif "RFE" in name:
        X_test = X_test_rfe
    elif "PCA" in name:
        X_test = X_test_pca
    else:
        X_test = X_test_all

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    results.add_row([name, acc, auc, f1, recall, precision])

# Print the evaluation results
print(results)

# Plot ROC curve for each model
fig, ax = plt.subplots()
for name, model in trained_models:
    if "kbest" in name:
        X_test = X_test_kbest
    elif "RFE" in name:
        X_test = X_test_rfe
    elif "PCA" in name:
        X_test = X_test_pca
    else:
        X_test = X_test_all

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')

# Plot the line representing the "Chance" level performance
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

# Set labels and title
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")

# Place legend at the lower right

ax.legend(loc="lower right")
plt.show()
