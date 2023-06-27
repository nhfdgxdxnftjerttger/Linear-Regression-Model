# Linear regression model using gradient descent

import numpy as np

def gradient_descent(X, y, learning_rate, num_iterations):
    n = len(y)
    theta = np.zeros(X.shape[1])
    for _ in range(num_iterations):
        y_pred = np.dot(X, theta)
        error = y_pred - y
        gradient = np.dot(X.T, error) / n
        theta -= learning_rate * gradient
    return theta

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Add bias term to the feature matrix
X = np.c_[np.ones((100, 1)), X]

# Perform gradient descent
learning_rate = 0.01
num_iterations = 1000
theta = gradient_descent(X, y, learning_rate, num_iterations)

# Print the learned parameters
print("Intercept:", theta[0])
print("Slope:", theta[1])
