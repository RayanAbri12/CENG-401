import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download and load the UCI Adult dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
raw_data = pd.read_csv(dataset_url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
raw_data = raw_data.dropna().drop_duplicates()

# Split into train, validation, and test sets
train_data, temp_data = train_test_split(raw_data, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print sizes
print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# Visualize the splits
sizes = [len(train_data), len(val_data), len(test_data)]
labels = ['Train', 'Validation', 'Test']
plt.bar(labels, sizes, color=['blue', 'orange', 'green'])
plt.title('Dataset Split Sizes')
plt.ylabel('Number of Samples')
plt.show()

# Calculate sizes and percentages
sizes = [len(raw_data), len(train_data), len(val_data), len(test_data)]
labels = ['Total', 'Train', 'Validation', 'Test']
colors = ['gray', 'blue', 'orange', 'green']
percentages = [size / sizes[0] * 100 for size in sizes]

plt.figure(figsize=(8,6))
bars = plt.bar(labels, sizes, color=colors)
plt.title('Dataset and Split Sizes')
plt.ylabel('Number of Samples')
plt.xlabel('Dataset Split')

# Annotate bars with counts and percentages
for i, (bar, size, pct) in enumerate(zip(bars, sizes, percentages)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{size}\n({pct:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.ylim(0, max(sizes)*1.15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Print useful information
print(f"Total samples: {sizes[0]}")
print(f"Train set: {sizes[1]} samples ({percentages[1]:.1f}%)")
print(f"Validation set: {sizes[2]} samples ({percentages[2]:.1f}%)")
print(f"Test set: {sizes[3]} samples ({percentages[3]:.1f}%)")
print("\nTrain set is used for model fitting.")
print("Validation set is used for hyperparameter tuning and model selection.")
print("Test set is used for final model evaluation.")
