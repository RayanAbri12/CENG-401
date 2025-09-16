# Decision Tree Classification on Wine Dataset

This project demonstrates a complete machine learning workflow for multi-class classification using the Wine dataset and Decision Tree algorithm. Each step is explained in detail for educational purposes.

## Steps

### 1. Download the Dataset

- **Script:** `download_wine.py`
- **Action:** Loads the Wine dataset from sklearn and saves it as CSV for reference.
- **Details:** The Wine dataset contains 178 samples, 13 features, and 3 classes (types of wine).

### 2. Preprocess the Data

- **Script:** `preprocess_wine.py`
- **Action:** Checks for missing values and standardizes features (important for many ML algorithms).
- **Details:** Standardization ensures all features have mean 0 and variance 1, improving model performance.

### 3. Split the Dataset

- **Script:** `split_wine.py`
- **Action:** Splits the data into train (80%) and test (20%) sets, stratified by class, and saves splits as CSV.
- **Details:** Stratified splitting maintains class balance in train and test sets, which is crucial for fair evaluation.

### 4. Train and Evaluate Decision Tree

- **Script:** `decision_tree_wine.py`
- **Action:** Trains a Decision Tree classifier on the training set, evaluates on the test set, prints accuracy and classification report.
- **Details:**
  - Decision Trees are non-parametric models that learn rules from data features.
  - The classification report includes precision, recall, f1-score for each class, as well as macro and weighted averages.
  - Accuracy is the proportion of correct predictions.
  - Macro avg is the unweighted mean of metrics for all classes.
  - Weighted avg is the mean weighted by the number of samples per class.

## How to Run

1. Run each script in order:
   ```bash
   python DecisionTree/download_wine.py
   python DecisionTree/preprocess_wine.py
   python DecisionTree/split_wine.py
   python DecisionTree/decision_tree_wine.py
   ```
2. Review printed outputs and CSV files for each step.

## Technical Notes

- **Decision Tree Hyperparameters:** You can tune parameters like `max_depth`, `min_samples_split`, and `criterion` to improve performance or control overfitting.
- **Feature Importance:** Decision Trees can show which features are most important for classification.
- **Visualization:** You can visualize the tree using `sklearn.tree.plot_tree` for better understanding.
- **Evaluation:** Always check both accuracy and the classification report to understand model strengths and weaknesses.

## Next Steps

- Try tuning Decision Tree hyperparameters.
- Compare Decision Tree results with other classifiers (e.g., KNN, SVM).

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
