import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)
data = data.dropna().drop_duplicates()

# Encode categorical columns
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Scale numeric columns
numeric_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

print("Transformed data (first 5 rows):")
print(data.head())
