import pandas as pd

# Column names from UCI repository documentation
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Load the raw data
raw_data = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)

print("First 5 rows of the labeled dataset:")
print(raw_data.head())
print("\nData shape:", raw_data.shape)
print("\nMissing values per column:")
print(raw_data.isnull().sum())
