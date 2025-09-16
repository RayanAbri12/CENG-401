# Naive Bayes Classification on Titanic Dataset

This project demonstrates a complete machine learning workflow for binary classification using the Titanic dataset and Naive Bayes algorithm. Each step is explained in detail for educational purposes.

## Steps

### 1. Download the Dataset

- **Script:** `download_titanic.py`
- **Action:** Loads the Titanic dataset from seaborn and saves it as CSV for reference.
- **Details:** The Titanic dataset contains information about passengers, including whether they survived (binary target).

### 2. Preprocess the Data

- **Script:** `preprocess_titanic.py`
- **Action:** Checks for missing values, encodes categorical variables, and standardizes features.
- **Details:** Standardization ensures all features have mean 0 and variance 1, improving model performance. Categorical variables are encoded to numeric values.

### 3. Split the Dataset

- **Script:** `split_titanic.py`
- **Action:** Splits the data into train (80%) and test (20%) sets, stratified by class, and saves splits as CSV.
- **Details:** Stratified splitting maintains class balance in train and test sets, which is crucial for fair evaluation.

### 4. Train and Evaluate Naive Bayes

- **Script:** `naive_bayes_titanic.py`
- **Action:** Trains a Gaussian Naive Bayes classifier on the training set, evaluates on the test set, prints accuracy and classification report.
- **Details:**
  - Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming feature independence.
  - The classification report includes precision, recall, f1-score for each class, as well as macro and weighted averages.
  - Accuracy is the proportion of correct predictions.
  - Macro avg is the unweighted mean of metrics for all classes.
  - Weighted avg is the mean weighted by the number of samples per class.

## How to Run

1. Run each script in order:
   ```bash
   python NaiveBayes/download_titanic.py
   python NaiveBayes/preprocess_titanic.py
   python NaiveBayes/split_titanic.py
   python NaiveBayes/naive_bayes_titanic.py
   ```
2. Review printed outputs and CSV files for each step.

## Technical Notes

- **Naive Bayes Assumptions:** Assumes features are independent and normally distributed (GaussianNB).
- **Evaluation:** Always check both accuracy and the classification report to understand model strengths and weaknesses.

## Next Steps

- Try tuning preprocessing steps or using other Naive Bayes variants (e.g., BernoulliNB, MultinomialNB).
- Compare Naive Bayes results with other classifiers (e.g., Logistic Regression, Random Forest).

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
