# Random Forest Classification on California Housing Dataset

This project demonstrates a complete machine learning workflow for multi-class classification using the California Housing dataset and Random Forest algorithm. Each step is explained in detail for educational purposes.

## Steps

### 1. Download the Dataset

- **Script:** `download_california_housing.py`
- **Action:** Loads the California Housing dataset from sklearn and saves it as CSV for reference.
- **Details:** The California Housing dataset contains 20,640 samples, 8 features (housing attributes), and the target is binned into 3 classes (Low, Medium, High median house value).

### 2. Preprocess the Data

- **Script:** `preprocess_california_housing.py`
- **Action:** Checks for missing values and standardizes features (important for many ML algorithms).
- **Details:** Standardization ensures all features have mean 0 and variance 1, improving model performance.

### 3. Split the Dataset

- **Script:** `split_california_housing.py`
- **Action:** Splits the data into train (80%) and test (20%) sets, stratified by class, and saves splits as CSV.
- **Details:** Stratified splitting maintains class balance in train and test sets, which is crucial for fair evaluation.

### 4. Train and Evaluate Random Forest

- **Script:** `random_forest_california_housing.py`
- **Action:** Trains a Random Forest classifier on the training set, evaluates on the test set, prints accuracy and classification report.
- **Details:**
  - Random Forest is an ensemble method that builds multiple decision trees and averages their predictions for better accuracy and robustness.
  - The classification report includes precision, recall, f1-score for each class, as well as macro and weighted averages.
  - Accuracy is the proportion of correct predictions.
  - Macro avg is the unweighted mean of metrics for all classes.
  - Weighted avg is the mean weighted by the number of samples per class.

## How to Run

1. Run each script in order:
   ```bash
   python RandomForest/download_california_housing.py
   python RandomForest/preprocess_california_housing.py
   python RandomForest/split_california_housing.py
   python RandomForest/random_forest_california_housing.py
   ```
2. Review printed outputs and CSV files for each step.

## Technical Notes

- **Random Forest Hyperparameters:** You can tune parameters like `n_estimators`, `max_depth`, and `min_samples_split` to improve performance or control overfitting.
- **Feature Importance:** Random Forests can show which features (housing attributes) are most important for classification.
- **Evaluation:** Always check both accuracy and the classification report to understand model strengths and weaknesses.

## Next Steps

- Try tuning Random Forest hyperparameters.
- Compare Random Forest results with other classifiers (e.g., Decision Tree, SVM).

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
