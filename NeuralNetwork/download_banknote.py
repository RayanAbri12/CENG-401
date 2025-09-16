# Step 1: Download and load the Banknote Authentication dataset (binary classification)
# The dataset is available from UCI repository
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(url, header=None, names=columns)

# Save to CSV for reference
df.to_csv('NeuralNetwork/banknote.csv', index=False)
print('Banknote Authentication dataset downloaded and saved as NeuralNetwork/banknote.csv')
