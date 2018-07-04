from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()


############## define Variables ###############
in_units = 784 ## 28 * 28
h1_units = 300 ## for this problem, 200 -1000, make no difference

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))

W2 = tf.Variable(tf.zeros([h1_units, 10])) ## output is 10 
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32) ## dropout ratio, less than 1 when train, eq 1 when predict


############ define graph ######################
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

########### define loss ########################
## cross entropy
y_ = tf.placeholder(tf.float32, [None, 10]) 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

############ train ############################

tf.global_variables_initializer().run()
epochs = 3000
batch_size = 100
for i in range(epochs):
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

########### test, accuracy ####################
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## accuracy.run(x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0)


feed_dict_test = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
loss, acc_test = sess.run([cross_entropy, accuracy], feed_dict=feed_dict_test)

## accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print ("Test loss: {:.2f}, accuracy: {:.01%}".format(loss, acc_test))

