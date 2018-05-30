import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print ("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print ("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print ("- Validation-set:\t{}".format(len(mnist.validation.labels)))

