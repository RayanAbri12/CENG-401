import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
columns = [
    'ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]
data = pd.read_csv('wdbc.data', header=None, names=columns)

# Drop ID column (not useful for ML)
data = data.drop('ID', axis=1)

# Encode Diagnosis (M=1, B=0)
le = LabelEncoder()
data['Diagnosis'] = le.fit_transform(data['Diagnosis'])

# Scale features
features = data.columns[1:]
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

print(data.head())
