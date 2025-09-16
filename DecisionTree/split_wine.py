# Step 3: Split the Wine dataset into train and test sets
# - Use 80% for training, 20% for testing
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('DecisionTree/wine_scaled.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Train set size: {X_train.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

# Save splits for reference
X_train.to_csv('DecisionTree/X_train.csv', index=False)
X_test.to_csv('DecisionTree/X_test.csv', index=False)
y_train.to_csv('DecisionTree/y_train.csv', index=False)
y_test.to_csv('DecisionTree/y_test.csv', index=False)
