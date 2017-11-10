"""
Good to use when the number of features are less than 1000.
Extremely efficient than gradient decent when above condition is satisfied
"""

import numpy as np

np.random.seed(111)

'''
The data is generated adding noise to the values from  y = 1 + 0.5x1 + 3x2 -2x3 equation
The expectation of the auto encoder is to get the values  w0, w1, w2 and w3  closer to 1, 0.5, 3 and -2  respectively
'''

'''generate random x1, x2, x3 values'''
x1 = np.random.random((1, 50))[0]
x2 = np.random.random((1, 50))[0]
x3 = np.random.random((1, 50))[0]

'''define x0 as 1. To be multiplied with w0 values'''
x0 = np.ones((1, 50))[0]

'''create X_train which is has x1, x2, x3 as columns and 50 rows'''
X = np.c_[x0, x1, x2, x3]

'''get the reference y value'''
y_reference = 1 + 0.5*x1 + 3*x2 -2*x3

'''add noise to the reference y value'''
y = y_reference + np.sqrt(0.01) * np.random.random((1, 50))[0]

'''Get X transpose'''
X_t = X.transpose()

'''Normal Equation'''
W = np.linalg.pinv(X_t.dot(X)).dot(X_t).dot(y)

print('\nNormal Equation optimization completed')
print('W Expected : [1, 0.5, 3, -2]' + '  Learned : ' + str(W))
