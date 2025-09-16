import pandas as pd

# Download the UCI Breast Cancer Wisconsin (Diagnostic) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
output_path = "wdbc.data"

# Download the file
pd.DataFrame(pd.read_csv(url, header=None)).to_csv(output_path, index=False, header=False)
print(f"Downloaded dataset to {output_path}")
