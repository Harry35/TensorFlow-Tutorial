import tensorflow as tf

hw = tf.constant("Hello World ! I love TensorFlow !")

sess = tf.Session()

print sess.run(hw)

sess.close()
