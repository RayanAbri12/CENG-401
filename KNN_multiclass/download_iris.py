# Step 1: Download and load the Iris dataset (multi-class classification)
# The Iris dataset is available directly from sklearn
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to CSV for reference
df.to_csv('KNN_multiclass/iris.csv', index=False)
print('Iris dataset downloaded and saved as KNN_multiclass/iris.csv')
