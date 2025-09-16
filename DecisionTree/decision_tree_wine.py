# Step 4: Run Decision Tree classifier on the Wine dataset (multi-class)
# - Train Decision Tree on training set
# - Evaluate on test set
# - Print accuracy and classification report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Load train and test splits
X_train = pd.read_csv('DecisionTree/X_train.csv')
X_test = pd.read_csv('DecisionTree/X_test.csv')
y_train = pd.read_csv('DecisionTree/y_train.csv').values.ravel()
y_test = pd.read_csv('DecisionTree/y_test.csv').values.ravel()

# Initialize and train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
# Step-by-step: Predict test set labels
# Step-by-step: Calculate accuracy and print classification report

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Test Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize the trained decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X_train.columns, class_names=['Class 0', 'Class 1', 'Class 2'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
