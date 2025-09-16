# KNN Multi-Class Classification on Iris Dataset

This project demonstrates a step-by-step machine learning workflow for multi-class classification using the Iris dataset and KNN algorithm.

## Steps

### 1. Download the Dataset

- **Script:** `download_iris.py`
- **Action:** Loads the Iris dataset from sklearn and saves it as CSV for reference.

### 2. Preprocess the Data

- **Script:** `preprocess_iris.py`
- **Action:** Checks for missing values and standardizes features (important for KNN).

### 3. Split the Dataset

- **Script:** `split_iris.py`
- **Action:** Splits the data into train (80%) and test (20%) sets, stratified by class, and saves splits as CSV.

### 4. Train and Evaluate KNN

- **Script:** `knn_iris.py`
- **Action:** Trains a KNN classifier (k=5) on the training set, evaluates on the test set, prints accuracy and classification report.

## Technical Details

- **Preprocessing:** Standardization ensures all features contribute equally to distance-based algorithms like KNN.
- **Stratified Splitting:** Maintains class balance in train and test sets.
- **KNN:** k=5 is a common default; you can tune this parameter for better results.
- **Evaluation:** Accuracy and classification report show model performance for each class.

## How to Run

1. Run each script in order:
   ```bash
   python KNN_multiclass/download_iris.py
   python KNN_multiclass/preprocess_iris.py
   python KNN_multiclass/split_iris.py
   python KNN_multiclass/knn_iris.py
   ```
2. Review printed outputs and CSV files for each step.

## Next Steps

- Try tuning the number of neighbors (`n_neighbors`) or using cross-validation for better results.
- Compare KNN results with other classifiers (e.g., SVM, Decision Tree).

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
