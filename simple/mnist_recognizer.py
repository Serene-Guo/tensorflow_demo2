import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from layer_utils import fc_layer
from mnist_class_plot import plot_images
from mnist_class_plot import plot_example_errors

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print ("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print ("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print ("- Validation-set:\t{}".format(len(mnist.validation.labels)))

logs_path= "../logs/mnistRecognizer"
learning_rate = 0.001
epochs = 3
batch_size = 100
display_freq = 100

img_h = img_w = 28
img_size_flat = img_h * img_w

n_classes = 10
h1 = 200 ## num of units in the first hidden layer


############## Define graph ############################
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

init = tf.global_variables_initializer()

############### Start Train ###############
sess = tf.InteractiveSession()
sess.run(init)
train_writer = tf.summary.FileWriter(logs_path, sess.graph)
num_tr_iter = int(mnist.train.num_examples / batch_size)

for epoch in range(epochs):
	print ("Training epoch: {}".format(epoch + 1))
	for iteration in range(num_tr_iter):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		
		feed_dict_batch = {x: batch_x, y: batch_y}
		sess.run(optimizer, feed_dict=feed_dict_batch)
		
		if iteration % display_freq == 0:
			loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
			print ("iter {}:\t loss={:.2f}, \tTraining Accuracy={:.01%}".format(iteration, loss_batch, acc_batch))
	
	### after each epoch, run validation
	feed_dict_valid = {x: mnist.validation.images, y: mnist.validation.labels}
	loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
	print ("---------------------------------------------------------------")
	print ("Epoch: {}, validation loss: {:.2f}, validation accuracy: {:.01%}".format(epoch, loss_valid, acc_valid))
	print ("---------------------------------------------------------------")



############### plot test ##################
feed_dict_test = {x: mnist.test.images, y:mnist.test.labels}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print ("------------------------------")
print ("Test loss: {:.2f}, test accuracy: {:.01%}".format(loss_test, acc_test))
print ("--------------------------------")

cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
cls_true = np.argmax(mnist.test.labels, axis=1)

#plot_images(mnist.test.images, cls_true, cls_pred, title="ALL Test DATA")
plot_example_errors(mnist.test.images, cls_true, cls_pred, title="Correct Example")
plot_example_errors(mnist.test.images, cls_true, cls_pred, title="Mis classified Examples", print_right=False)

sess.close()

