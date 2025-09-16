# Step 1: Download and load the Fashion MNIST dataset (multi-class classification)
# The Fashion MNIST dataset is available from torchvision
import torch
from torchvision import datasets, transforms
import pandas as pd

# Download and load Fashion MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.FashionMNIST(root='./NeuralNetwork', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./NeuralNetwork', train=False, download=True, transform=transform)

# Save a small sample to CSV for reference
X_train = train_set.data[:1000].reshape(1000, -1).numpy()
y_train = train_set.targets[:1000].numpy()
X_test = test_set.data[:200].reshape(200, -1).numpy()
y_test = test_set.targets[:200].numpy()

train_df = pd.DataFrame(X_train)
train_df['target'] = y_train
train_df.to_csv('NeuralNetwork/fashion_mnist_train.csv', index=False)
test_df = pd.DataFrame(X_test)
test_df['target'] = y_test
test_df.to_csv('NeuralNetwork/fashion_mnist_test.csv', index=False)
print('Fashion MNIST sample downloaded and saved as CSV.')
