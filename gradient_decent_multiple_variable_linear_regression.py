import numpy as np

np.random.seed(111)

'''
The data is generated adding noise to the values from  y = 1 + 0.5x1 + 3x2 -2x3 equation
The expectation of the auto encoder is to get the values w1, w2, w3 and b closer to 0.5, 3, -2 and 1 respectively
'''

'''generate random x1, x2, x3 values'''
x1 = np.random.random((1, 50))[0]
x2 = np.random.random((1, 50))[0]
x3 = np.random.random((1, 50))[0]

'''create X_train which is has x1, x2, x3 as columns and 50 rows'''
X_train = np.c_[x1, x2, x3]

'''get the reference y value'''
y_reference = 1 + 0.5*x1 + 3*x2 -2*x3


'''add noise to the reference y value'''
y_train = y_reference + np.sqrt(0.01)*np.random.random((1, 50))[0]

'''W is a row vector of the shape [w1, w2, w3]'''
W = np.random.random((1, 3))[0]
b = np.random.random((1, 1))[0]

'''number of training examples'''
m = len(x1)

'''parameters'''
learning_rate = 0.1
epochs = 500


def gradient_descent(X, y):
    global W, b, learning_rate, epochs
    for _epoch in range(epochs):
        hypothesis = X.dot(W) + b

        '''cost function'''
        cost = np.divide(1, 2*m) * np.sum((hypothesis-y) ** 2)
        print(cost)

        '''partial derivatives of the cost function with respect to W and b'''
        gradient_w = np.divide(1, m) * X.transpose().dot(hypothesis-y)
        gradient_b = np.divide(1, m) * np.sum(hypothesis-y)

        '''calculating new W and b values simultaneously'''
        temp_w = W - learning_rate*gradient_w
        temp_b = b - learning_rate*gradient_b

        '''updating W and b simultaneously'''
        W = temp_w
        b = temp_b

        print('\nepoch ' + str(_epoch) + '  W : ' + str(W) + '  b : ' + str(b) + ' Cost : ' + str(cost))

'''send data to the gradient optimizer to optimize values for W and b'''
gradient_descent(X_train, y_train)

print('\nGradient optimization completed')
print('W Expected : [0.5, 3, -2]' + '  Learned : ' + str(W))
print('b Expected : 1.0' + '  Learned : ' + str(b))