# Step 2: Preprocess the Titanic dataset
# - Check for missing values
# - Standardize features
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('NaiveBayes/titanic.csv')

# Check for missing values
def check_missing(df):
    missing = df.isnull().sum()
    print('Missing values per column:')
    print(missing)
check_missing(df)

# Standardize features (excluding target 'survived')
features = df.drop('survived', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Create new DataFrame with scaled features
scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
scaled_df['survived'] = df['survived']
scaled_df.to_csv('NaiveBayes/titanic_scaled.csv', index=False)
print('Features standardized and saved as NaiveBayes/titanic_scaled.csv')
