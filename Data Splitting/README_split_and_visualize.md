# Dataset Splitting and Visualization Demo

This project demonstrates how to split a real-world dataset into training, validation, and test sets, and visualize the sizes of each split using a bar chart. The example uses the UCI Adult (Census Income) dataset, a common benchmark in machine learning.

## What Does This Code Do?

- **Downloads and loads the UCI Adult dataset** directly from the UCI Machine Learning Repository.
- **Cleans the data** by removing missing and duplicate entries.
- **Splits the dataset** into three parts:
  - **Training set (60%)**: Used to fit the machine learning model.
  - **Validation set (20%)**: Used to tune model hyperparameters and select the best model.
  - **Test set (20%)**: Used to evaluate the final model's performance on unseen data.
- **Visualizes the sizes** of the total dataset, training, validation, and test sets in a single bar chart, with both sample counts and percentages annotated.
- **Prints useful information** about each split and their roles in the machine learning workflow.

## How to Run

1. Make sure you have Python 3, pandas, scikit-learn, and matplotlib installed.
2. Run the script:
   ```bash
   python split_and_visualize_dataset.py
   ```
   (If using a virtual environment, use the appropriate python command.)

## Example Output

- **Bar chart** showing the number and percentage of samples in each split.
- **Printed summary**:
  - Total samples: 32,537
  - Train set: 19,522 samples (60.0%)
  - Validation set: 6,507 samples (20.0%)
  - Test set: 6,508 samples (20.0%)
- **Explanation of each split's purpose**:
  - Train set: Used for model fitting.
  - Validation set: Used for hyperparameter tuning and model selection.
  - Test set: Used for final model evaluation.

## Why Is This Important?

- **Proper dataset splitting** prevents overfitting and ensures that your model generalizes well to new, unseen data.
- **Visualization** helps students understand the distribution and importance of each split.
- **Clear workflow** for teaching and practical application in machine learning courses.

## Customization

- You can adjust the split ratios by changing the `test_size` and `random_state` parameters in the code.
- The code can be adapted to other datasets by changing the download URL and column names.

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
