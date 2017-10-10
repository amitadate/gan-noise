import tensorflow as tf 
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mnist import cnn_model
from mnist import softmax_loss
from mnist import l2_loss
from mnist import adam_optimizer
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Some stats about the MNIST dataset
print('--------------------')
print("Training-set: " + str(len(mnist.train.labels)))
print("Test-set: " + str(len(mnist.test.labels)))
print("Validation-set: " + str(len(mnist.validation.labels)))

# Width of the input image 
mnist_img_width = 28
# Height of the input iamge
mnist_img_height = 28
#number of channels
mnist_img_num_channel = 1
# Number of the classes (0-9)
num_classes = 10
#The data in the dataset is stored in one dimensional array of size image_width X image_height
mnist_flat_image = mnist_img_width * mnist_img_height
#number of results to show
number_of_rows = 10
#Set the adversarial noise limit, the noise will be between [delta, -delta]
delta = 0.5
# Number of training steps for MNIST classification
minst_train_step = 1000
# Number of training steps for Adversarial Noise
noise_train_step = 1000
# Batch size per step
batch_size = 64
# The weight of L2 Loss. Since adverserial loss is a combantion of two losses, softmax and l2.
# This paramter sets how important the L2 loss comparing with the softmax loss
l2_weight = 0.01

#Placeholder for the input image/s
#The shape of X is [None, minst_flat_image]
#None: means the placeholder can take any number of images
x = tf.placeholder(tf.float32, shape=[None, mnist_flat_image], name='x')

#Placeholder for the true value of the input image/s
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')

#Dropout parameter for regularization 
#During training it will be set to 0.5
#During testing it will be set to 1.0
#See below
keep_prob = tf.placeholder(tf.float32)

#Since the input is one dimensional array of size image_width X image_height
#The CNN expects the input is an image
#We reshape the input data to [image_width, image_height, number_of_channels]
x_image = tf.reshape(x, [-1, mnist_img_width, mnist_img_height,mnist_img_num_channel])

#Variable for the adversarial noise
#The shape of the noise image will be the same as the input image [image_width, image_height, number_of_channels]
#This variable will not be trained along with the rest of the variable when we train MNIST dataset for
#classification. Therefore, trainable will be set to false
noise_img = tf.Variable(tf.zeros([mnist_img_width, mnist_img_height, mnist_img_num_channel]),
                      name='noise_img', trainable=False)

#The adversarial noise will be clipped to the give delta
noise_img_clip = tf.assign(noise_img, tf.clip_by_value(noise_img, -delta, delta))

#Add the noise to the input image
#Note: The input image will not be affected in the first optimizatin since the noise will be zero
x_noisy_image = x_image + noise_img

#We want to make sure the image values are still in range. 
#Therefore, The image is clipped between [0, 1]
x_noisy_image = tf.clip_by_value(x_noisy_image, 0.0, 1.0)

#Calculate logists of in the input image/s
logits = cnn_model(x_noisy_image, keep_prob)

#Compute softmax cross entropy loss 
softmax_loss = softmax_loss(logits, y)

#Perform one step optimization using Adam optimization to minimize the softmax loss
softmax_mnist_optimizer = adam_optimizer(softmax_loss, lr=1e-4)

#Compute the L2 loss given the noise image
l2_noise_loss = l2_loss(noise_img, l2_weight)

#Add the softmax loss and l2 loss for the adversarial noise optimization
combined_loss = softmax_loss + l2_noise_loss

#Perform one step optimization to minimize the combined loss
#Since this optimization does not update all variable in the network, we need to 
#Tell the optimizer what variable should be optimized via var_list
noise_optimizer = adam_optimizer(combined_loss, lr=1e-3, var_list=[noise_img])

#Vector of boolean to show if the predicted class equals the true class of an image/s
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

#Calculate the accuracy of the model
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create a tensorflow session to execute the graph and train the models
sess = tf.Session()

#Initialize all variables but the noise variable
sess.run(tf.global_variables_initializer())

#Initialize the adversarial variable
sess.run(tf.variables_initializer([noise_img]))

#Train the CNN with minst_train_step iteration
#You can get an accuracy of 99.2% if you train the model for 20,000 iterations
#It may take some time depends if you are using GPU or CPU
for i in range(minst_train_step):
    #Extract a batch from the training set
	x_batch, y_batch = mnist.train.next_batch(batch_size)
    #Check if the number of iteration is i % 100 so we can test the model
	if i % 100 == 0:
        #Calculate the accuracy of the current batch
		train_acc = sess.run([accuracy], feed_dict={x:x_batch, y: y_batch, keep_prob: 1.0})
		print("Step " + str(i) + " , Train Acc: " + str(train_acc))
    #Execute one step of optimization give the x_batch, and y_batch
	sess.run([softmax_mnist_optimizer], feed_dict={x:x_batch, y:y_batch, keep_prob:0.5})

#Now the model is trained 
#Test the model using the testing dataset
test_acc = sess.run([accuracy], feed_dict={x: mnist.test.images, 
	y: mnist.test.labels, keep_prob: 1.0})
print("Test Acc: " + str(test_acc))

#Initialize the adversarial variable
sess.run(tf.variables_initializer([noise_img]))

#Train the adversarial noise with noise_train_step
#You can get the noise when performing 1000 iterations 
for i in range(noise_train_step):
    #Extract a batch from the training set
	x_batch, y_batch = mnist.train.next_batch(batch_size)
    
    #Find the images of number 2 from the batch
	samples_indx = np.where(y_batch[:, 2] == 1)[0]

    #Update the true values of those images to 6
	y_batch[samples_indx, 2] = 0
	y_batch[samples_indx, 6] = 1

    #Check if the batch does not have images of number 2
	if x_batch[samples_indx, :].shape[0] == 0:
		continue

    #Execute one step of optimization give the x_batch, and the updated y_batch
	sess.run([noise_optimizer], feed_dict={x:x_batch[samples_indx, :], y:y_batch[samples_indx, :], keep_prob:0.5})
    #Make sure the new values of the adversarial noise are within range
	sess.run(noise_img_clip)

    #print a message every 100 iterations
	if i % 100 == 0:
		print("Done with " + str(i) + " iterations")

#Create a softmax layer to get the rank of each label
softmax = tf.nn.softmax(logits)

#Variable to keep track of how many images with label 2 have been seen
count = 0 

#Create a plot of 10X3 
f, axarr = plt.subplots(number_of_rows, 3, figsize=(9, 30))

#Iterate over the training set to find 10 images with label 2
#It can be done without while loop only using numpy operation but I want 
#In each run you get different samples of images with label 2

while True:
    #get an image from the training set 
    x_batch, y_batch = mnist.train.next_batch(1)
    
    #check if the image is 2
    if np.argmax(y_batch) != 2:
        continue
    
    #Since the input is one dimensional array of size image_width X image_height
    #The CNN expects the input is an image
    #We reshape the input data to [image_width, image_height, number_of_channels]
    image = x_batch.reshape((28, 28, 1))
    
    #Get the value of the noise from the graph
    noise = sess.run(noise_img)
    
    #Combine the input image with the loss
    image_with_noise = image + noise
    
    #Make sure the image is within range [0, 1]
    image_with_noise = np.clip(image_with_noise, 0.0, 1.0)

    #Subtract the noise from the image so when we feed the image with noise to the network
    #It will be the original image
    image_without_noise = image - noise
    
    #Reshape the image without noise from [image_width, image_height] to [1, image_width * image_height]
    image_without_noise = image_without_noise.reshape((1, mnist_flat_image))

    #Get the probability of the image with noise 
    probabilites_with_noise = sess.run([softmax], feed_dict={x:x_batch, keep_prob:1.0})
    
    #Get the probability of the original image
    probabilites_without_noise = sess.run([softmax], feed_dict={x:image_without_noise, keep_prob:1.0})

    #Plot the results along with the probability
    axarr[count, 0].imshow(image[:, :, 0], cmap='binary', interpolation='nearest')
    axarr[count, 0].set_title('True Label: ' + str(np.argmax(y_batch)) + '\nPredicted Label: ' + str(np.argmax(probabilites_without_noise)) +
                              '\n Prob: ' + str(np.max(probabilites_without_noise)))
    axarr[count, 1].imshow(np.squeeze(noise), interpolation='nearest', cmap='seismic',vmin=-1.0, vmax=1.0)
    axarr[count, 1].set_title('Noise with delta = ' + str(delta))
    axarr[count, 2].imshow(image_with_noise[:, :, 0], cmap='binary', interpolation='nearest')
    axarr[count, 2].set_title('Predicted Label WITH Noise: ' + str(np.argmax(probabilites_with_noise)) +
                       '\n Prob: ' + str(np.max(probabilites_with_noise)))
    
    count += 1
    #Check if the number of images that have been seen equals the number of rows
    if count == number_of_rows:
        break
#Adjust the plot spacing 
f.subplots_adjust(hspace=0.5)
#Save the plot in the same directory
f.savefig('2_to_6_with_delta_' + str(delta) +'.png')   # save the figure to file
#Show plot
plt.show()