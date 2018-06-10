import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.linspace(-7, 7, 180)

def sigmoid(inputs):
	y = [1 / float(1 + np.exp(-x)) for x in inputs]
	return y

def relu(inputs):
	y = [x * (x > 0) for x in inputs]
	return y

def tanh(inputs):
	y = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) - np.exp(-x)) for x in inputs]
	return y

def softplus(inputs):
	y = [np.log(1 + np.exp(x)) for x in inputs]
	return y

y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

sess = tf.Session()

y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

plt.figure(1, figsize=(8, 6))

plt.suplot(221)
plt.plot(x, y_sigmoid, c='red', label='Sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
