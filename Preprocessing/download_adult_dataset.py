# Download the UCI Adult dataset (Census Income)
import urllib.request

# URL of the raw data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
output_path = "adult.data"

# Download the file
urllib.request.urlretrieve(url, output_path)
print(f"Downloaded dataset to {output_path}")
