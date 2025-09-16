# Machine Learning & Artificial Intelligence: Data Preprocessing Demo

This project demonstrates practical data preprocessing steps on a real-world raw dataset (UCI Adult/Census Income). Each step is implemented in Python and explained below.

## Dataset

- **Source:** [UCI Machine Learning Repository - Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
- **File:** `adult.data` (downloaded automatically)

## Step-by-Step Guide

### 1. Data Acquisition

- **Script:** `download_adult_dataset.py`
- **Description:** Downloads the raw dataset from the UCI repository and saves it as `adult.data`.
- **How to run:**
  ```bash
  python3 download_adult_dataset.py
  ```

### 2. Data Labeling & Loading

- **Script:** `preprocess_adult_dataset.py`
- **Description:** Loads the raw data into a pandas DataFrame and assigns meaningful column names.
- **How to run:**
  ```bash
  python3 preprocess_adult_dataset.py
  ```

### 3. Data Cleaning

- **Script:** `clean_adult_dataset.py`
- **Description:** Removes rows with missing values and duplicate entries to ensure data quality.
- **How to run:**
  ```bash
  python3 clean_adult_dataset.py
  ```

### 4. Data Transformation

- **Script:** `transform_adult_dataset.py`
- **Description:**
  - Encodes categorical variables into numeric values using LabelEncoder.
  - Scales numeric features using StandardScaler.
- **How to run:**
  ```bash
  python3 transform_adult_dataset.py
  ```

## Requirements

- Python 3.6+
- pandas
- scikit-learn

## Notes

- Each script prints sample output to help you understand the effect of each preprocessing step.

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
