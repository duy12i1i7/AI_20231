import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from polyreg import PolynomialRegression, learningCurve


def plotLearningCurve(errorTrain, errorTest, regLambda, degree):
    '''
    Plot computed learning curve
    '''
    minX = 3

    # Check for NaN or Inf in errorTest and handle it
    if np.any(np.isnan(errorTest)) or np.any(np.isinf(errorTest)):
        print("Error in errorTest: Contains NaN or Inf")
        return

    maxY = max(errorTest[minX+1:])

    plt.plot(errorTrain, 'r-o', label='Training Error')
    plt.plot(errorTest, 'b-o', label='Testing Error')
    plt.axhline(y=1, color='k', linestyle='--')
    plt.legend(loc='best')
    plt.title(f'Learning Curve (d={degree}, lambda={regLambda})')
    plt.xlabel('Training samples')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.ylim((0, maxY))
    plt.xlim((minX, 10))


def generateLearningCurve(X, y, degree, regLambda):
    '''
    Computing learning curve via leave one out CV
    '''
    loo = LeaveOneOut()
    errorTrains, errorTests = [], []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        errTrain, errTest = learningCurve(X_train, y_train, X_test, y_test, regLambda, degree)
        errorTrains.append(errTrain)
        errorTests.append(errTest)

    errorTrain = np.mean(errorTrains, axis=0)
    errorTest = np.mean(errorTests, axis=0)
    plotLearningCurve(errorTrain, errorTest, regLambda, degree)


if __name__ == "__main__":
    '''
    Main function to test polynomial regression
    '''
    # load the data
    filePath = "data/polydata.dat"
    allData = np.loadtxt(filePath, delimiter=',')

    X = allData[:, 0]
    y = allData[:, 1]

    plt.figure(figsize=(15, 10))
    degrees = [1, 4, 8]
    lambdas = [0, 0.1, 1, 100]

    for i, degree in enumerate(degrees):
        for j, regLambda in enumerate(lambdas):
            plt.subplot(len(degrees), len(lambdas), i*len(lambdas) + j + 1)
            generateLearningCurve(X, y, degree, regLambda)

    plt.tight_layout()
    plt.show()
