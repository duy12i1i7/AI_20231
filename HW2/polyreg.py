'''
Template for polynomial regression
AUTHOR Eric Eaton, Xiaoxiang Hu
'''
import numpy as np

class PolynomialRegression:

    def __init__(self, degree=1, regLambda=1E-8):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        self.weights = None

    def polyfeatures(self, X, degree):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if (np.max(X) - np.min(X)) < 1e-10:  
            print("Warning: Range of X is too small for normalization.")
            X = X - np.min(X)
        else :
            X = (X - np.min(X)) / (np.max(X) - np.min(X))
        n = X.shape[0]
        X_poly = np.zeros((n, degree + 1))
        X_poly[:, 0] = 1  # bias term
        for d in range(degree):
            X_poly[:, d + 1] = X[:, 0] ** (d + 1)

        return X_poly

    def fit(self, X, y):

        X_poly = self.polyfeatures(X, self.degree)

        regMatrix = self.regLambda * np.eye(X_poly.shape[1])
        regMatrix[0, 0] = 0  # Do not regularize the bias term

        # Normal equation
        self.weights = np.linalg.pinv(X_poly.T.dot(X_poly) + regMatrix).dot(X_poly.T).dot(y)

    def predict(self, X):

        X_poly = self.polyfeatures(X, self.degree)
        return X_poly.dot(self.weights)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset;
        errorTrain[i] = np.multiply(err, err).mean();
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest;
        errorTest[i] = np.multiply(err, err).mean();
    
    return (errorTrain, errorTest)