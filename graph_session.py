import tensorflow as tf

const1 = tf.constant([[2,2]])
const2 = tf.constant([[4],[4]])

multi = tf.matmul(const1, const2)

sess = tf.Session()
result = sess.run(multi)

print(result)

sess.close()

with tf.Session() as sess:
	result2 = sess.run(multi)
	print("Ok: %s", result2)
