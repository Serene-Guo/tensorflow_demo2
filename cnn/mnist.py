from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

## W: [5,5,1,32], 5*5, 1 channel, 32 kernels
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



############### define variable ##################
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32]) ### 5*5, channel is 1, kernels is 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  ###h_conv1 is 28 * 28  * 32, 32 kernels.

####### 2nd conv
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
## h_pool2, 7*7*64
######## full connection 
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
### drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

############### softmax 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#################### cross entropy 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


############### start training ###################
tf.global_variables_initializer().run()

epochs = 20000
batch_size = 50
print_freq = 100
for i in range(epochs):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	
	if i % print_freq == 0:
		eval_feed_dict = {x: batch_x, y_: batch_y, keep_prob: 1.0}	
		train_acc = accuracy.eval(feed_dict = eval_feed_dict)
		print ("Epoch: {}, acc: {:.01%}".format(i, train_acc))
	
	train_feed_dict = {x: batch_x, y_: batch_y, keep_prob: 0.5}
	train_step.run(feed_dict=train_feed_dict)

############## test 
test_x = mnist.test.images
test_y = mnist.test.labels
keep_prob = 1.0
test_feed_dict = {x: test_x, y_: test_y, keep_prob: 1.0}

print ("Test, accuracy: %g" % accuracy.eval(feed_dict=test_feed_dict))
