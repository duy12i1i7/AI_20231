import numpy as np
from scipy.optimize import fmin_tnc  # Used for optimizing the cost function

class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-Z))
        '''
        return 1 / (1 + np.exp(-Z))

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
        reg = (regLambda/(2*m)) * (theta[1:].T @ theta[1:])
        return cost + reg

    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        grad = (1/m) * (X.T @ (h - y))
        grad[1:] = grad[1:] + (regLambda/m) * theta[1:]
        return grad

    def fit(self, X, y):
        '''
        Trains the model
        '''
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        self.theta = np.zeros(X.shape[1])

        result = fmin_tnc(func=self.computeCost, x0=self.theta, fprime=self.computeGradient, args=(X, y, self.regLambda))
        self.theta = result[0]

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        '''
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        probabilities = self.sigmoid(X @ self.theta)
        return [1 if p >= 0.5 else 0 for p in probabilities]

