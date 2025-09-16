# Step 1: Download and load the Wine dataset (multi-class classification)
# The Wine dataset is available directly from sklearn
from sklearn.datasets import load_wine
import pandas as pd

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Save to CSV for reference
df.to_csv('DecisionTree/wine.csv', index=False)
print('Wine dataset downloaded and saved as DecisionTree/wine.csv')
