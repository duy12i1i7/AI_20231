import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from mapFeature import mapFeature
from logreg import LogisticRegression

if __name__ == "__main__":
    # Load Data
    filename = 'data/data2.dat'
    data = loadtxt(filename, delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]  # Changed to a 1D array

    # Standardize the data
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Map features into a higher dimensional feature space
    X = mapFeature(X[:, 0], X[:, 1])

    # Train logistic regression
    logregModel = LogisticRegression()
    logregModel.fit(X, y)

    # Reload the data for 2D plotting purposes
    data = loadtxt(filename, delimiter=',')
    PX = data[:, 0:2]
    y = data[:, 2]

    # Standardize the data
    mean = PX.mean(axis=0)
    std = PX.std(axis=0)
    PX = (PX - mean) / std

    # Plot the decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = PX[:, 0].min() - 0.5, PX[:, 0].max() + 0.5
    y_min, y_max = PX[:, 1].min() - 0.5, PX[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    allPoints = np.c_[xx.ravel(), yy.ravel()]
    allPoints = mapFeature(allPoints[:, 0], allPoints[:, 1])
    Z = logregModel.predict(allPoints)

    # Put the result into a color plot
    Z = np.array(Z).reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')

    # Plot the training points
    plt.scatter(PX[:, 0], PX[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    # Configure the plot display
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks([])
    plt.yticks([])

    plt.show()
