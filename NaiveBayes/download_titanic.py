# Step 1: Download and load the Titanic dataset (binary classification)
# The Titanic dataset is available from seaborn
import seaborn as sns
import pandas as pd

# Load dataset
titanic = sns.load_dataset('titanic')

# Select relevant columns and drop rows with missing values
cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df = titanic[cols].dropna()

# Encode categorical variables
# 'sex' and 'embarked' to numeric
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])
df['embarked'] = le_embarked.fit_transform(df['embarked'])

# Save to CSV for reference
df.to_csv('NaiveBayes/titanic.csv', index=False)
print('Titanic dataset downloaded and saved as NaiveBayes/titanic.csv')
