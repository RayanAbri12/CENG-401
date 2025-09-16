# Step 1: Download and load the California Housing dataset (regression, but can be used for classification by binning target)
# The California Housing dataset is available directly from sklearn
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# For classification, bin the target into categories (low, medium, high)
df['target_class'] = pd.qcut(df['target'], q=3, labels=['Low', 'Medium', 'High'])

# Save to CSV for reference
df.to_csv('RandomForest/california_housing.csv', index=False)
print('California Housing dataset downloaded and saved as RandomForest/california_housing.csv')
