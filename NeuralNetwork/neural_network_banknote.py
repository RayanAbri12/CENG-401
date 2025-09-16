# Step 4: Build and train a small neural network on the Banknote dataset using PyTorch
# - Use gradient descent (Adam optimizer)
# - Binary classification
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load train and test splits
X_train = pd.read_csv('NeuralNetwork/X_train.csv').values
X_test = pd.read_csv('NeuralNetwork/X_test.csv').values
y_train = pd.read_csv('NeuralNetwork/y_train.csv').values.ravel()
y_test = pd.read_csv('NeuralNetwork/y_test.csv').values.ravel()

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define a small neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/50, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()
    acc = accuracy_score(y_test.numpy(), y_pred_class.numpy())
    print(f'Neural Network Test Accuracy: {acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test.numpy(), y_pred_class.numpy()))
