import tensorflow as tf
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

'''define parameters'''
training_epochs = 1000
learning_rate = 0.001

'''define nodes in tensorflow graph'''
X = tf.placeholder(tf.float32, name='X')
y = tf.placeholder(tf.float32, name='y')
W = tf.Variable(np.random.randn(), name='W')
b = tf.Variable(np.random.randn(), name='b')

'''define activation - linear activation is considered'''
activation = tf.add(tf.multiply(X, W), b)

'''define cost function'''
cost = tf.reduce_mean(tf.pow(activation-y, 2))

'''define optimizer - gradient decent'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

'''define global initializer'''
init = tf.global_variables_initializer()

'''start a tensorflow session'''
with tf.Session() as sess:

    '''initialize variables'''
    sess.run(init)

    '''run optimizer on each training example for defined epochs'''
    for epoch in range(training_epochs):
        for (_x, _y) in zip(X_train, y_train):
            sess.run(optimizer, feed_dict={X: _x, y: _y})

        if epoch % 100 == 0:
            print('\nepoch ' + str(epoch) + ' : W = ' + str(sess.run(W)) + ' b = ' + str(sess.run(b)))

    print('\nLinear Regression model training completed')
    print('W Expected : 0.8' + '  Learned : ' + str(sess.run(W)))
    print('b Expected : 2.0' + '  Learned : ' + str(sess.run(b)))


