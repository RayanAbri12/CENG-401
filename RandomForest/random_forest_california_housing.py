# Step 4: Run Random Forest classifier on the California Housing dataset (multi-class)
# - Train Random Forest on training set
# - Evaluate on test set
# - Print accuracy and classification report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load train and test splits
X_train = pd.read_csv('RandomForest/X_train.csv')
X_test = pd.read_csv('RandomForest/X_test.csv')
y_train = pd.read_csv('RandomForest/y_train.csv').values.ravel()
y_test = pd.read_csv('RandomForest/y_test.csv').values.ravel()

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
# Step-by-step: Predict test set labels
# Step-by-step: Calculate accuracy and print classification report

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Test Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize feature importances
importances = rf.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title('Random Forest Feature Importances')
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()
