# Step 4: Run KNN classifier on the Iris dataset (multi-class)
# - Train KNN on training set
# - Evaluate on test set
# - Print accuracy and classification report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load train and test splits
X_train = pd.read_csv('KNN_multiclass/X_train.csv')
X_test = pd.read_csv('KNN_multiclass/X_test.csv')
y_train = pd.read_csv('KNN_multiclass/y_train.csv').values.ravel()
y_test = pd.read_csv('KNN_multiclass/y_test.csv').values.ravel()

# Initialize and train KNN (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
# Step-by-step: Predict test set labels
# Step-by-step: Calculate accuracy and print classification report

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Test Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
