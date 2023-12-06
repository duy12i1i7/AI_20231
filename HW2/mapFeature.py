import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.

    Returns a new feature array with polynomial features up to the 6th power.

    Arguments:
        x1: an n-dimensional array
        x2: an n-dimensional array
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    degree = 6
    output = np.ones((x1.shape[0], 1))  # Start with a column of ones for the intercept term

    for i in range(1, degree + 1):
        for j in range(i + 1):
            terms = (x1 ** (i - j)) * (x2 ** j)
            output = np.hstack((output, terms.reshape(-1, 1)))

    return output
