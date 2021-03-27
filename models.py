import numpy as np


class LinearRegression:
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.theta = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias = 0
        self.theta = np.zeros((n_features, 1))

        for _ in range(self.n_iters):
            y_pred = self.bias + np.dot(self.theta, X.T)

            dw = np.dot((y_pred - y), X) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.theta -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        return self.bias + np.dot(self.theta, X.T)


class LogisticRegression:
    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.theta = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias = 0
        self.theta = np.zeros((n_features, 1))

        for _ in range(self.n_iters):
            linear_model = self.bias + np.dot(self.theta, X.T)
            y_pred = self._sigmoid(linear_model)

            dw = np.dot((y_pred - y), X) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.theta -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        linear_model = self.bias + np.dot(self.theta, X.T)
        y_pred = self._sigmoid(linear_model)[0]
        return [1 if i > 0.5 else 0 for i in y_pred]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
