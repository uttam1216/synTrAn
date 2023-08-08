import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
dataset = pd.read_csv('data/StaticMap/merged_updated_train.csv')
print(dataset['user_encoded'].value_counts())
# Separate features (X) and target variable (y)
X = dataset.drop('user_encoded', axis=1)  # Exclude the user_id column
y = dataset['user_encoded']

# Create an instance of RandomOverSampler
oversampler = RandomOverSampler(random_state=42)

# Perform random oversampling
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Create a new balanced dataset
balanced_dataset = pd.concat([y_resampled, X_resampled], axis=1)

# Print the class distribution in the new balanced dataset
print(balanced_dataset['user_encoded'].value_counts())
print(balanced_dataset)
balanced_dataset.to_csv('data/StaticMap/merged_updated_train_oversampled.csv', index=False)