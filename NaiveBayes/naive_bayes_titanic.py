# Step 4: Run Naive Bayes classifier on the Titanic dataset (binary classification)
# - Train Naive Bayes on training set
# - Evaluate on test set
# - Print accuracy and classification report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load train and test splits
X_train = pd.read_csv('NaiveBayes/X_train.csv')
X_test = pd.read_csv('NaiveBayes/X_test.csv')
y_train = pd.read_csv('NaiveBayes/y_train.csv').values.ravel()
y_test = pd.read_csv('NaiveBayes/y_test.csv').values.ravel()

# Initialize and train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict and evaluate
# Step-by-step: Predict test set labels
# Step-by-step: Calculate accuracy and print classification report

y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Test Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
