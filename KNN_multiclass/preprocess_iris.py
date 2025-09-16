# Step 2: Preprocess the Iris dataset
# - Check for missing values
# - Standardize features
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('KNN_multiclass/iris.csv')

# Check for missing values
def check_missing(df):
    missing = df.isnull().sum()
    print('Missing values per column:')
    print(missing)
check_missing(df)

# Standardize features (excluding target)
features = df.drop('target', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create new DataFrame with scaled features
scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
scaled_df['target'] = df['target']
scaled_df.to_csv('KNN_multiclass/iris_scaled.csv', index=False)
print('Features standardized and saved as KNN_multiclass/iris_scaled.csv')
