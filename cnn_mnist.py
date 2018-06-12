import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

input_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
output_y = tf.placeholder(tf.int32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

conv1 = tf.layers.conv2d(
	inputs=input_x_image,
	filter=32,
	kernel_size=[5,5],
	strides=1,
	padding='same',
	activation=tf.nn.relu
)
