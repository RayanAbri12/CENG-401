import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
raw_data = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Data cleaning: drop rows with missing values
data_cleaned = raw_data.dropna()

# Remove duplicates (if any)
data_cleaned = data_cleaned.drop_duplicates()

print("Cleaned data shape:", data_cleaned.shape)
print("Missing values after cleaning:")
print(data_cleaned.isnull().sum())
