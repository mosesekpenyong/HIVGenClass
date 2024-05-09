import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
df = pd.read_excel('/content/drive/My Drive/PythonCodeFiles/Dataset.xlsx', sheet_name=0)

# Drop the target column named 'Drg_rel'
X = df.drop('Drg_rel', axis=1)
y = df['Drg_rel']

# Initialize StratifiedKFold for nested resampling
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Initialize Borderline-SMOTE with a specific sampling strategy based on class counts
borderline_smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, kind='borderline-1')

# Initialize SMOTE for both training and test data
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=list(X.columns) + ['Drg_rel', 'DatasetType'])

# Encode string columns for training data
label_encoder = LabelEncoder()
for column in X.select_dtypes(include='object').columns:
    X[column] = label_encoder.fit_transform(X[column])

# Apply Borderline-SMOTE within each fold of cross-validation
for fold_index, (train_index, test_index) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()

    # Resample the training data using Borderline-SMOTE within each fold
    X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train, y_train)

    # Add 'DatasetType' column to indicate it's training data
    X_train_resampled['DatasetType'] = 'Train'
    X_train_resampled['Drg_rel'] = y_train_resampled

    # Encode string columns for testing data using the label encoder fitted on training data
    for column in X_test.select_dtypes(include='object').columns:
        X_test[column] = label_encoder.transform(X_test[column])

    # Resample the test data using SMOTE
    X_test_resampled, y_test_resampled = smote.fit_resample(X_test, y_test)

    # Add 'DatasetType' column to indicate it's testing data
    X_test_resampled['DatasetType'] = 'Test'
    X_test_resampled['Drg_rel'] = y_test_resampled

    # Append the results to the DataFrame
    results_df = pd.concat([results_df, X_train_resampled, X_test_resampled], axis=0)

# Save the results to an Excel file
results_df.to_excel('/content/drive/My Drive/PythonCodeFiles/ResamplingResults_StratifiedKFold_Modified.xlsx', index=False)
