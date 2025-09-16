import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and preprocess the data
columns = [
    'ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]
data = pd.read_csv('wdbc.data', header=None, names=columns)
data = data.drop('ID', axis=1)

# Step 2: Encode Diagnosis (M=1, B=0)
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
data['Diagnosis'] = le.fit_transform(data['Diagnosis'])

# Step 3: Scale features
features = data.columns[1:]
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Step 4: Split into train, validation, and test sets
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 5: Train SVM on training set
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Step 6: Evaluate on validation set
val_pred = svm.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")
print("Validation Classification Report:")
print(classification_report(y_val, val_pred))

# Step 7: Evaluate on test set
test_pred = svm.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print("Test Classification Report:")
print(classification_report(y_test, test_pred))

# Each step is explained in comments above for teaching purposes.
