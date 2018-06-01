import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

#from layer_utils import weight_variable
#from layer_utils import bias_variable
from layer_utils import fc_layer
from plot_utils import plot_image

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print ("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print ("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print ("- Validation-set:\t{}".format(len(mnist.validation.labels)))

logs_path= "../logs/noiseRemoval"
learning_rate = 0.001
epochs = 3
batch_size = 100
display_freq = 100

# network parameters
img_h = img_w = 28

# image are staroed in one-dimensional arrays of this length
img_size_flat = img_h * img_w

# number of units in the hidden layer
h1 = 100

# level of the noise in noisy data
noise_level = 0.6




with tf.variable_scope("Input"):
	x_original = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X_original')
	x_noisy = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X_noisy')

fc1 = fc_layer(x_noisy, h1, 'Hidden_layer', use_relu=True)
out = fc_layer(fc1, img_size_flat, "Output_layer", use_relu=False)

# define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
	with tf.variable_scope('Loss'):
		loss = tf.reduce_mean(tf.losses.mean_squared_error(x_original, out), name = 'loss')
		tf.summary.scalar('loss', loss)
	with tf.variable_scope('Optimizer'):
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'Adam-op').minimize(loss)


# Initializing the variables
init = tf.global_variables_initializer()

# Add 5 images from original, noisy and reconstructed samples to summaries
tf.summary.image('original', tf.reshape(x_original, (-1, img_w, img_h, 1)), max_outputs=5)
tf.summary.image('noisy', tf.reshape(x_noisy, (-1, img_w, img_h, 1)), max_outputs=5)
tf.summary.image('reconstruced', tf.reshape(out, (-1, img_w, img_h, 1)), max_outputs=5)


# Merge all the summaries
merged = tf.summary.merge_all()

## Launch the graph (session)
sess = tf.InteractiveSession() # using InteractiveSession instead of Session to test network in separate cell
sess.run(init)
train_writer = tf.summary.FileWriter(logs_path, sess.graph)
num_tr_iter = int(mnist.train.num_examples / batch_size)
global_step = 0
for epoch in range(epochs):
	print ('Training epoch:{}'.format(epoch + 1))
	for iteration in range(num_tr_iter):
		batch_x, _ = mnist.train.next_batch(batch_size)
		batch_x_noisy = batch_x + noise_level * np.random.normal(loc=0.0, scale = 1.0, size=batch_x.shape)
		global_step += 1 
		## Run optimization op (back prop)
		feed_dict_batch = {x_original: batch_x, x_noisy: batch_x_noisy}
		_, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
		train_writer.add_summary(summary_tr, global_step)
		if iteration % display_freq == 0:
			loss_batch = sess.run(loss, feed_dict=feed_dict_batch)
			print ("iter {0:3d}:\t Reconstruction loss={1:.3f}".format(iteration, loss_batch))
	
	## Run validation after each epoch
	x_valid_original = mnist.validation.images
	x_valid_noisy = x_valid_original + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_valid_original.shape)
	
	feed_dict_valid = {x_original: x_valid_original, x_noisy: x_valid_noisy}
	loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
	print ('----------------------------------------------------')
	print ("Epoch: {}, validation loss: {:.3f}".format(epoch+1, loss_valid))
	print ('----------------------------------------------------')

	

### Test the network after training
# Make a noisy image
test_samples = 5
x_test = mnist.test.images[:test_samples]
x_test_noisy = x_test + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

#Reconstruct a clean image from noisy image
x_reconstruct = sess.run(out, feed_dict={x_noisy: x_test_noisy})

# Calculate the loss between reconstructed image and original image
loss_test = sess.run(loss, feed_dict={x_original: x_test, x_noisy:x_test_noisy})
print ("-------------------------------------------------------")
print ("Test loss of original image compared to reconstructed image: {:.3f}".format(loss_test))
print ("-------------------------------------------------------")

###### Plot images
plot_image(x_test, x_test_noisy, x_reconstruct)

sess.close()



