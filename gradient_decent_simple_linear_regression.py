import numpy as np

np.random.seed(111)

'''
The data is generated adding noise to the values from  y = 0.8x + 2 equation
Therefore the expectation of the auto encoder is to get the values w and b closer to 0.8 and 2 respectively
'''

'''generate random x values'''
X_train = np.random.random((1, 50))[0]

'''get the reference y value'''
y_reference = 0.8*X_train + 2

'''add noise to the reference y value'''
y_train = y_reference + np.sqrt(0.01)*np.random.random((1, 50))[0]

W = np.random.random()
b = np.random.random()

'''number of training examples'''
m = len(X_train)

'''parameters'''
learning_rate = 0.01
epochs = 5000


def gradient_descent(X, y):
    global W, b, learning_rate, epochs
    for _epoch in range(epochs):
        hypothesis = W*X + b

        '''cost function'''
        cost = np.divide(1, 2*m) * np.sum((hypothesis-y) ** 2)
        print(hypothesis)
        exit()

        '''partial derivatives of the cost function with respect to W and b'''
        gradient_w = np.divide(1, m) * np.sum((hypothesis-y)*X)
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
print('W Expected : 0.8' + '  Learned : ' + str(W))
print('b Expected : 2.0' + '  Learned : ' + str(b))