'''
    TEST SCRIPT FOR POLYNOMIAL REGRESSION
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "data/polydata.dat"
    try:
        allData = np.loadtxt(filePath, delimiter=',')
    except IOError:
        print(f"Error: File {filePath} not found.")
        exit(1)

    X = allData[:, 0]
    y = allData[:, 1]
    # X=  [ 0.   0.5  1.   2.   3.   5.   6.   7.   8.   9.  10. ] 
    # y= [2.  2.5 2.5 5.  4.  3.9 5.  4.5 4.2 4.  2. ]
    X = X[:, np.newaxis]
    d = 8
    model = PolynomialRegression(degree=d, regLambda=0)
    model.fit(X, y)
    
    xpoints = np.linspace(np.min(X), np.max(X), 100).reshape(-1,1)
    # output predictions
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = '+str(d))
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
