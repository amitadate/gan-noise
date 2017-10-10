import tensorflow as tf 
import numpy as np

# Create Weights variable of a given shape
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Create a bais variable of a given shape
def bais_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#This model is inspired by the official tensorflow tutorial
#I made some changes to create scops when necessary to make the visualization easier if you want to use tensorbored
#https://www.tensorflow.org/tutorials/mnist/pros/
def cnn_model(image, keep_prob):
	#Create a conv layer of filter size 5X5, stride 1, and depth 32
	with tf.variable_scope('conv1') as scope:
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bais_variable([32])

		conv = tf.nn.conv2d(image, W_conv1, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, b_conv1)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #Create a max pooling layer of filter size 2X2, stride 1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
	                         padding='SAME', name='pool1')
	#Create a conv layer of filter size 5X5, stride 1, and depth 64
	with tf.variable_scope('conv2') as scope:
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bais_variable([64])

		conv = tf.nn.conv2d(pool1, W_conv2, [1, 1, 1, 1], padding='SAME')
		pre_activation = tf.nn.bias_add(conv, b_conv2)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #Create a max pooling layer of filter size 2X2, stride 1
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
	                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    #Create a fully connected of size [7 * 7 * 64, 1024]
	with tf.variable_scope('fc1') as scope:
		W_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bais_variable([1024])

		pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
		fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1, name=scope.name)
    #Create a drop out layer for regulization
	fc1_drop = tf.nn.dropout(fc1, keep_prob)
    
    #Create a fully connected of size [1024, 10]
	with tf.variable_scope('softmax_linear') as scope:
		W_fc2 = weight_variable([1024, 10])
		b_fc2 = bais_variable([10])

		softmax_linear = tf.add(tf.matmul(fc1_drop, W_fc2), b_fc2, name=scope.name)

	return softmax_linear

#calculate the softmax loss
def softmax_loss(logits, labels):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
	mean_cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
	return mean_cross_entropy

#calculate the L2 loss
def l2_loss(noise_image, l2_weight):
	return l2_weight * tf.nn.l2_loss(noise_image)

#Perform one step optimization for the given variable list using a given lr
def adam_optimizer(loss, lr=None, var_list=None):
	optimizer = tf.train.AdamOptimizer(lr).minimize(loss, var_list=var_list)
	return optimizer