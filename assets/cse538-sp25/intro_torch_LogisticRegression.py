import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the logistic regression model using nn.Module
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # one output neuron (for binary classification)
        self.linear.weight.data.fill_(0.0)    # initialize weights and the bias to zero
        self.linear.bias.data.fill_(0.0)
        print ("Model initialized with weights: ", self.linear.weight.data.numpy().round(2), self.linear.bias.data.numpy().round(2))
        
    def forward(self, x):
        # Use the sigmoid activation from nn.functional
        return torch.sigmoid(self.linear(x))

# Training loop using Batch Gradient Descent
def train_batch(X, y, num_epochs, learning_rate):
    model = LogisticRegressionModel(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    loss_list = []
    loss = criterion(model(X), y)  # Initial loss
    loss_list.append(loss.item())
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)             # Forward pass over the entire dataset
        loss = criterion(outputs, y)   # Compute normalized negative log likelihood
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights    
        loss_list.append(loss.item())
    return model, loss_list

# Training loop using Stochastic Gradient Descent (one sample at a time)
def train_stochastic(X, y, num_epochs, learning_rate):
    model = LogisticRegressionModel(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    loss_list = []
    N = X.size(0)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = criterion(model(X), y).item()
        loss_list.append(epoch_loss)    
        # Shuffle the data indices each epoch
        indices = torch.randperm(N)
        for i in indices:
            xi = X[i].unsqueeze(0)  # Make it a 2D tensor of shape (1, num_features)
            yi = y[i].unsqueeze(0)  # Shape (1, 1)
            
            optimizer.zero_grad()
            output = model(xi)
            loss = criterion(output, yi)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    epoch_loss = criterion(model(X), y).item()
    loss_list.append(epoch_loss)
    return model, loss_list

# Function to compute accuracy on a given dataset
def compute_accuracy(model, X, y):
    model.eval()  # set to evaluation mode
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    return accuracy

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # ----- Data Generation -----
    # Generate one-dimensional data for two classes:
    N_per_class = 100
    X0 = np.random.normal(loc=0.0, scale=1.0, size=(N_per_class, 1))
    X1 = np.random.normal(loc=1.0, scale=1.0, size=(N_per_class, 1))
    X_np = np.vstack((X0, X1))
    y_np = np.hstack((np.zeros(N_per_class), np.ones(N_per_class)))
    
    # Shuffle the dataset
    indices = np.arange(X_np.shape[0])
    np.random.shuffle(indices)
    X_np = X_np[indices]
    y_np = y_np[indices]
    
    # Convert to PyTorch tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
    
    # ----- Hyperparameters -----
    num_epochs = 200
    learning_rate = 0.1
    
    # ----- Batch Gradient Descent Training -----
    print("Training with Batch Gradient Descent:")
    torch.manual_seed(42)
    model_batch, loss_batch = train_batch(X, y, num_epochs, learning_rate)
    acc_batch = compute_accuracy(model_batch, X, y)
    final_loss_batch = loss_batch[-1]
    print("Final Negative Log Likelihood (Batch): {:.4f}".format(final_loss_batch))
    print("Accuracy (Batch): {:.2f}%".format(acc_batch * 100))
    print("Model weights: ", (model_batch.linear.weight.data.numpy().round(2), model_batch.linear.bias.data.numpy().round(2)))
    
    # ----- Stochastic Gradient Descent Training -----
    print("\nTraining with Stochastic Gradient Descent:")
    torch.manual_seed(42)
    model_sgd, loss_sgd = train_stochastic(X, y, num_epochs, learning_rate)
    acc_sgd = compute_accuracy(model_sgd, X, y)
    final_loss_sgd = loss_sgd[-1]
    print("Final Negative Log Likelihood (SGD): {:.4f}".format(final_loss_sgd))
    print("Accuracy (SGD): {:.2f}%".format(acc_sgd * 100))
    print("Model weights: ", (model_sgd.linear.weight.data.numpy().round(2), model_sgd.linear.bias.data.numpy().round(2)))
    
    # ----- Plotting the Loss Curves -----
    epochs = range(0, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_batch, label='Batch Gradient Descent')
    plt.plot(epochs, loss_sgd, label='Stochastic Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Negative Log Likelihood')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig("logistic_regression_loss_torch.png", dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    main()
