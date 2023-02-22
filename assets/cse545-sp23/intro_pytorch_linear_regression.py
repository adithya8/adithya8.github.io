import torch

# Set random seed for reproducibility
torch.manual_seed(42)

def get_data(n):
    # Generate some toy data
    x = torch.rand(n, 1)
    y = (2 * x + 0.5)
    return x, y

def perform_gradient_descent(x, y, w, b, lr, num_iters):
    # Perform gradient descent
    for i in range(num_iters):
        # Calculate model predictions
        y_pred = torch.matmul(x, w) + b
        
        # Calculate mean squared error
        loss = torch.mean((y_pred - y) ** 2)
        
        # Calculate gradients
        loss.backward()
        
        # Update parameters
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            
            # Manually zero gradients
            w.grad.zero_()
            b.grad.zero_()
            
        # Print progress every 100 iterations
        if (i) % 200 == 0:
            print(f"Iteration {i + 1}: Loss = {loss.item():.4f}")
            print (f" w = {w.item():.3f}, b = {b.item():.3f}")
            
    print(f"Iteration {i + 1}: Loss = {loss.item():.4f}")
    print (f" w = {w.item():.3f}, b = {b.item():.3f}")

if __name__ == "__main__":
    x, y = get_data(1000)

    # Define model parameters
    w = torch.randn(x.shape[-1], 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    # Define learning rate and number of iterations
    lr = 0.1
    num_iters = 1000
    
    perform_gradient_descent(x, y, w, b, lr, num_iters)