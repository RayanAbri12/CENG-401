# Neural Network Classification on Banknote Authentication Dataset

This project demonstrates a complete machine learning workflow for binary classification using the Banknote Authentication dataset and a small neural network built with PyTorch. Each step is explained in detail for educational purposes.

## Steps

### 1. Download the Dataset

- **Script:** `download_banknote.py`
- **Action:** Loads the Banknote Authentication dataset from the UCI repository and saves it as CSV for reference.
- **Details:** The dataset contains 1372 samples, 4 features (variance, skewness, curtosis, entropy), and a binary target (authentic or not).

### 2. Preprocess the Data

- **Script:** `preprocess_banknote.py`
- **Action:** Checks for missing values and standardizes features.
- **Details:** Standardization ensures all features have mean 0 and variance 1, improving model performance.

### 3. Split the Dataset

- **Script:** `split_banknote.py`
- **Action:** Splits the data into train (80%) and test (20%) sets, stratified by class, and saves splits as CSV.
- **Details:** Stratified splitting maintains class balance in train and test sets, which is crucial for fair evaluation.

### 4. Build and Train Neural Network

- **Script:** `neural_network_banknote.py`
- **Action:** Defines a small neural network in PyTorch, trains it using gradient descent (Adam optimizer), evaluates on the test set, and prints accuracy and classification report.
- **Details:**
  - The network has one hidden layer (4 inputs → 8 hidden units → 1 output).
  - Uses ReLU activation and sigmoid for binary output.
  - Trained for 50 epochs using binary cross-entropy loss.
  - The classification report includes precision, recall, f1-score for each class, as well as macro and weighted averages.
  - Accuracy is the proportion of correct predictions.

## How to Run

1. Run each script in order:
   ```bash
   python NeuralNetwork/download_banknote.py
   python NeuralNetwork/preprocess_banknote.py
   python NeuralNetwork/split_banknote.py
   python NeuralNetwork/neural_network_banknote.py
   ```
2. Review printed outputs and CSV files for each step.

## Technical Notes

- **Neural Network Architecture:** You can modify the number of layers and units for experimentation.
- **Optimizer:** Adam is used for efficient gradient descent.
- **Evaluation:** Always check both accuracy and the classification report to understand model strengths and weaknesses.

## Next Steps

- Try tuning network architecture or training parameters.
- Compare neural network results with other classifiers (e.g., Naive Bayes, Random Forest).

---

**Prepared for CENG 401: Machine Learning and Artificial Intelligence**
