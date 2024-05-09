import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import time
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
df = pd.read_excel('/content/drive/My Drive/PythonCodeFiles/Dataset.xlsx', sheet_name=10)

# Assume the target variable is in the column named 'Mut_sp'
X = df.drop('Mut_sp', axis=1)
y = df['Mut_sp']
dataset_type = df['DatasetType']  # Assuming the column name is 'DatasetType'

# Change 'Train' to 1 and 'Test' to 2
df['DatasetType'] = df['DatasetType'].replace({'Train': 1, 'Test': 2})

# Split the data into train and test sets based on 'DatasetType'
X_train = X[df['DatasetType'] == 1]
y_train = y[df['DatasetType'] == 1]
X_test = X[df['DatasetType'] == 2]
y_test = y[df['DatasetType'] == 2]

# Define the SVM classifier
svm_classifier = SVC(probability=True)

# Define metrics
metrics = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']

# Define a function to perform GridSearchCV and output results
def perform_grid_search(classifier, param_grid, metrics, X_train, y_train, X_test, y_test):
    print(f"Performing GridSearchCV for {classifier.__class__.__name__}...")

    grid_search = GridSearchCV(classifier, param_grid, cv=10, scoring=metrics, refit='accuracy')

    start_train_time = time.time()
    grid_search.fit(X_train, y_train)
    end_train_time = time.time()

    start_test_time = time.time()
    # Predictions on the test set
    y_test_pred = grid_search.predict(X_test)
    end_test_time = time.time()

    # Predictions probabilities for ROC-AUC
    y_test_proba = grid_search.predict_proba(X_test)[:, 1]

    results = {}

    for metric in metrics:
        if metric == 'roc_auc':
            results[f'Train {metric}'] = roc_auc_score(y_train, grid_search.predict_proba(X_train)[:, 1])
            results[f'Test {metric}'] = roc_auc_score(y_test, y_test_proba)
        else:
            train_metric = globals()[f'{metric}_score'](y_train, grid_search.predict(X_train))
            test_metric = globals()[f'{metric}_score'](y_test, y_test_pred)

            results[f'Train {metric}'] = train_metric
            results[f'Test {metric}'] = test_metric

    results['Train Time (s)'] = end_train_time - start_train_time
    results['Test Time (s)'] = end_test_time - start_test_time

    return results

# Perform GridSearchCV for SVM
svm_results = perform_grid_search(svm_classifier, {}, metrics, X_train, y_train, X_test, y_test)

# Display the results
print("\nResults Summary:")
results_df = pd.DataFrame({'SVM': svm_results})
print(results_df)