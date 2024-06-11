import numpy as np


class SGD:
    def __init__(self, parameters, lr=1e-3):
        """
        Initialize Stochastic Gradient Descent
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.data -= parameter.grad.data * self.alpha

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.data *= 0
        

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.

        Parameters:
            parameters (list): List of parameters (weights) to optimize.
            lr (float): Learning rate (default: 0.001).
            beta1 (float): Exponential decay rate for the first moment estimates (default: 0.9).
            beta2 (float): Exponential decay rate for the second moment estimates (default: 0.999).
            epsilon (float): Small value to prevent division by zero (default: 1e-8).
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
        self.t = 0

    def zero_grad(self):
        """Zero the gradients of all parameters."""
        for param in self.parameters:
            param.grad.data[:] = 0

    def step(self):
        """Perform a single optimization step."""
        self.t += 1
        for i, param in enumerate(self.parameters):
            grad = param.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


