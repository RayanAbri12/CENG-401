# SVM Binary Classification: Breast Cancer Wisconsin Dataset

This project demonstrates step-by-step how to run a Support Vector Machine (SVM) for binary classification using the UCI Breast Cancer Wisconsin (Diagnostic) dataset. Each technical step is explained in code comments and in this README.

## Steps Overview

1. **Download the Dataset**

   - Source: [UCI ML Repository - Breast Cancer Wisconsin (Diagnostic)](<https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)>)
   - File: `wdbc.data` (downloaded automatically)

2. **Preprocessing**

   - Load data and assign column names.
   - Drop the ID column (not useful for ML).
   - Encode the target label (`Diagnosis`): Malignant (M) = 1, Benign (B) = 0.
   - Scale all feature columns for better SVM performance.

3. **Dataset Splitting**

   - Split into features (`X`) and target (`y`).
   - Use `train_test_split` to create:
     - Training set (60%)
     - Validation set (20%)
     - Test set (20%)

4. **Train SVM Classifier**

   - Use scikit-learn's `SVC` with a linear kernel.
   - Fit the model on the training set.

5. **Evaluate Model**
   - Predict and evaluate on the validation set (for tuning).
   - Predict and evaluate on the test set (for final performance).
   - Print accuracy and detailed classification reports (precision, recall, f1-score).

## How to Run

1. Make sure you have Python 3, pandas, scikit-learn installed.
2. Run the scripts in order:
   ```bash
   python download_breast_cancer_dataset.py
   python preprocess_breast_cancer.py
   python split_breast_cancer.py
   python svm_breast_cancer.py
   ```

## Example Output

- **Validation Accuracy:** ~97%
- **Test Accuracy:** ~96%
- **Classification Report:** Shows precision, recall, f1-score for each class (Malignant/Benign)

## Technical Notes

- **Label Encoding:** Converts categorical target to numeric for SVM.
- **Feature Scaling:** SVMs are sensitive to feature scales; standardization improves results.
- **Splitting:** Validation set is used for model selection/tuning, test set for unbiased evaluation.
- **SVM:** Linear kernel is used for simplicity and interpretability.

## Customization

- You can change the kernel type, adjust split ratios, or try other classifiers for experimentation.
- The workflow can be adapted to other binary classification datasets.

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
