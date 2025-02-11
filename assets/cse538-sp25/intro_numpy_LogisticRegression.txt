import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionWithSGD:
    def __init__(self, num_features):
        # Include a bias term: weights shape = (num_features + 1,)
        self.weights = np.zeros(num_features + 1)
    
    def add_bias(self, X):
        """Augment X with a column of ones for the bias term."""
        N = X.shape[0]
        return np.hstack((np.ones((N, 1)), X))
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X):
        X_bias = self.add_bias(X)
        return self.sigmoid(np.dot(X_bias, self.weights))
    
    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)
    
    def compute_loss(self, X, y):
        """
        Compute the normalized Negative Log Likelihood (NLL):
            J(β) = -1/N * sum[ y*log(p) + (1-y)*log(1-p) ]
        """
        N = X.shape[0]
        p = self.predict_proba(X)
        # Avoid log(0) issues by clipping the probabilities.
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
    
    def compute_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def train(self, X, y, num_iterations, alpha):
        """
        Train the model using full-batch gradient descent.
        This method uses the entire dataset for each gradient update.
        """
        N = X.shape[0]
        X_bias = self.add_bias(X)  # shape (N, num_features+1)
        losses = []
        for i in range(num_iterations):
            losses.append((i, self.compute_loss(X, y)))
            # Compute predicted probabilities for all samples.
            p = self.sigmoid(np.dot(X_bias, self.weights))
            # Compute the gradient of the normalized NLL.
            gradient = np.dot(X_bias.T, (p - y)) / N
            # Update the weights.
            self.weights -= alpha * gradient
        losses.append((num_iterations, self.compute_loss(X, y)))
        return losses
    
    def train_stochastic(self, X, y, num_epochs, alpha):
        """
        Train the model using true stochastic gradient descent (SGD).
        The update is performed one sample at a time.
        """
        N = X.shape[0]
        X_bias = self.add_bias(X)
        losses = []
        for epoch in range(num_epochs):
            # Loop over each sample (you can also shuffle here for better performance).
            losses.append((epoch, self.compute_loss(X, y)))
            for i in range(N):
                x_i = X_bias[i]  # a single sample (including bias)
                p_i = self.sigmoid(np.dot(x_i, self.weights))
                # Compute the gradient for this sample.
                gradient = (p_i - y[i]) * x_i
                # Update the weights.
                self.weights -= alpha * gradient
        losses.append((num_epochs, self.compute_loss(X, y)))
        return losses

def main():
    np.random.seed(42)
    
    # Generate dataset:
    #  - Two classes with one-dimensional features.
    #  - Class 0: samples from N(0, 1)
    #  - Class 1: samples from N(1, 1)
    N_per_class = 100
    X0 = np.random.normal(loc=0.0, scale=1.0, size=(N_per_class, 1))
    X1 = np.random.normal(loc=1.0, scale=1.0, size=(N_per_class, 1))
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(N_per_class), np.ones(N_per_class)))
    
    # Shuffle the dataset:
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Hyperparameters
    iterations = 200   # number of iterations for batch training
    epochs = 200       # number of epochs for SGD (each epoch processes all samples once)
    alpha = 0.1        # learning rate
    
    # ----- Batch Gradient Descent Training -----
    print("Batch gradient descent training:")
    model_batch = LogisticRegressionWithSGD(num_features=1)
    losses_bgd = model_batch.train(X, y, num_iterations=iterations, alpha=alpha)
    loss_batch = model_batch.compute_loss(X, y)
    accuracy_batch = model_batch.compute_accuracy(X, y)
    print("Negative Log Likelihood (Batch): {:.4f}".format(loss_batch))
    print("Accuracy (Batch): {:.2f}%".format(accuracy_batch * 100))
    print ("Weights: ", model_batch.weights.round(2))
    
    # ----- Stochastic Gradient Descent Training -----
    print("\nStochastic gradient descent training:")
    model_sgd = LogisticRegressionWithSGD(num_features=1)
    losses_sgd = model_sgd.train_stochastic(X, y, num_epochs=epochs, alpha=alpha)
    loss_sgd = model_sgd.compute_loss(X, y)
    accuracy_sgd = model_sgd.compute_accuracy(X, y)
    print("Negative Log Likelihood (SGD): {:.4f}".format(loss_sgd))
    print("Accuracy (SGD): {:.2f}%".format(accuracy_sgd * 100))
    print ("Weights: ", model_sgd.weights.round(2))
    
    # ----- Plot the losses -----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(*zip(*losses_bgd), label="Batch GD", color='blue', linestyle='-', marker='o', markersize=3)
    ax.plot(*zip(*losses_sgd), label="Stochastic GD", color='red', linestyle='-', marker='x', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Likelihood")
    ax.legend()
    plt.savefig("logistic_regression_loss.png", dpi=600, bbox_inches='tight')    

if __name__ == "__main__":
    main()
