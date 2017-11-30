import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(111)

'''
The data is generated adding noise to the values from  y = 0.8x + 2 equation
Therefore the expectation of the auto encoder is to get the values w and b closer to 0.8 and 2 respectively
'''

# generate random x values
X_train = np.random.random((1, 50))[0]

# get the reference y value
y_reference = 0.8 * X_train + 2

# add noise to the reference y value
y_train = y_reference + np.sqrt(0.01) * np.random.random((1, 50))[0]

# reshape (model expect a 2D array and X_train is a 1D. So reshape (50,) in to (50, 1))
X_train = X_train.reshape(-1, 1)

# build model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# get weights and bias of the learned model
w = model.coef_[0]
b = model.intercept_

print('\nSklearn Linear Regression completed')
print('w Expected : 0.8' + '  Learned : ' + str(w))
print('b Expected : 2.0' + '  Learned : ' + str(b))
