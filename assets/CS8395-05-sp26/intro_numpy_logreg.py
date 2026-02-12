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
    
    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)
    
    def compute_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

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
    epochs = 200       # number of epochs for SGD (each epoch processes all samples once)
    alpha = 0.1        # learning rate
        
    # ----- Stochastic Gradient Descent Training -----
    print("\nStochastic gradient descent training:")
    model_sgd = LogisticRegressionWithSGD(num_features=1)
    losses_sgd = model_sgd.train_stochastic(X, y, num_epochs=epochs, alpha=alpha)
    loss_sgd = model_sgd.compute_loss(X, y)
    accuracy_sgd = model_sgd.compute_accuracy(X, y)
    print("Negative Log Likelihood (SGD): {:.4f}".format(loss_sgd))
    print("Accuracy (SGD): {:.2f}%".format(accuracy_sgd * 100))
    print ("Weights: ", model_sgd.weights.round(2))

    # Hyperparameters
    epochs = 200       # number of epochs for SGD (each epoch processes all samples once)
    alpha = 0.01        # learning rate
        
    # ----- Stochastic Gradient Descent Training -----
    print("\nStochastic gradient descent training:")
    model_sgd2 = LogisticRegressionWithSGD(num_features=1)
    losses_sgd2 = model_sgd2.train_stochastic(X, y, num_epochs=epochs, alpha=alpha)
    loss_sgd2 = model_sgd2.compute_loss(X, y)
    accuracy_sgd2 = model_sgd2.compute_accuracy(X, y)
    print("Negative Log Likelihood (SGD): {:.4f}".format(loss_sgd2))
    print("Accuracy (SGD): {:.2f}%".format(accuracy_sgd2 * 100))
    print ("Weights: ", model_sgd2.weights.round(2))


    # ----- Plot the losses -----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(*zip(*losses_sgd), label="alpha = 0.1", color='red', linestyle='-', marker='x', markersize=3)
    ax.plot(*zip(*losses_sgd2), label="alpha = 0.01", color='blue', linestyle='-', marker='o', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Likelihood")
    ax.legend()
    plt.title("Logistic Regression Loss")
    plt.savefig("logistic_regression_loss_alpha_0.1_and_0.01.png", dpi=600, bbox_inches='tight')    

if __name__ == "__main__":
    main()