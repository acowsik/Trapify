"""
Training an auto encoder using a WAV format
Dependencies: Numpy and TensorFlow
make a directory /checkpoint/ in the cwd
"""
import os
import numpy as np
import tensorflow as tf
from autoencoder import AutoEncoder
from datagenerator import DataGenerator
from datetime import datetime
from progress.bar import Bar

# learning parameters
learning_rate = 0.002
num_epochs = 500
batch_size = 250
dropout_rate = 0

in_size = 1025
out_size = 1025

# list of size of three hidden layers
hid_layers = [512, 256, 128]

#training and testing paths
train_file = 'train1.wav'
val_file = 'test.wav'

# TF placeholders
x = tf.placeholder(tf.float32, [batch_size, in_size])
y = tf.placeholder(tf.float32, [batch_size, out_size])
keep_prob = tf.placeholder(tf.float32)

"""
NO GPU
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
"""

cwd = os.getcwd()
checkpoint_path = os.path.join(cwd, "checkpoint")
print("Checkpoints at " + checkpoint_path)

# Initialize model
model = AutoEncoder(in_size, out_size, hid_layers)

# variable representing the model output with sigmoid
score = tf.sigmoid(model.run(x, keep_prob))

saver = tf.train.Saver()

# GPU functions
"""with tf.device("/gpu:0"):"""
with tf.device("/cpu:0"):
	loss = tf.reduce_mean(tf.pow(y - score, 2))
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
	init = tf.global_variables_initializer()
		

# Initalize the data generator seperately for the training and validation set
train_generator = DataGenerator(train_file, out_size, in_size)
val_generator = DataGenerator(val_file, out_size, in_size)


# Get the number of training/validation steps per epoch
"""
FIX THIS
"""
train_batches_per_epoch = train_generator.length() // batch_size
val_batches_per_epoch = val_generator.length() // batch_size


with tf.Session() as sess:
	# initializing all variables
	sess.run(init)

	print("{} Start training...".format(datetime.now()))

	# looping over epochs
	for epoch in range(num_epochs):
	
		print ("{} Epoch number: {}".format(datetime.now(), epoch+1))
		step = 1

		train_cost = 0.
		train_count = 0

		for _ in Bar('Epoch ' + str(epoch), suffix="%(index)d/%(max)d %(eta)d").iter(range(train_batches_per_epoch)):

			batch_xs = train_generator.next_batch(batch_size)
			#running the train op
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_xs, keep_prob: dropout_rate})
			cost = sess.run(loss, feed_dict = {x: batch_xs, y: batch_xs, keep_prob: 1.})

			train_cost += cost
			train_count += 1

		train_cost /= train_count
		print("{} Training Cost = {:.4f}".format(datetime.now(), train_cost))


		# Testing
		print("{} Start validation".format(datetime.now()))
		test_cost = 0.
		test_count = 0

		for _ in Bar("Validation " + str(epoch), suffix="%(index)d/%(max)d %(eta)d").iter(range(val_batches_per_epoch)):
			batch_tx = val_generator.next_batch(batch_size)
			cost = sess.run(loss, feed_dict = {x: batch_tx, y: batch_tx, keep_prob: 1.})

			test_cost += cost
			test_count += 1

		test_cost /= test_count
		print("{} Test Cost = {:.4f}".format(datetime.now(), test_cost))


		# saving the end of the num of epochs
		if epoch == 1 or True:
			print("{} Saving checkpoint of model...".format(datetime.now()))

			#save checkpoint of the model
			try:
				checkpoint_name = os.path.join(checkpoint_path, 'my_model')
				print(checkpoint_name, 'checkpoint name')
				save_path = saver.save(sess, checkpoint_name)
			except:
				try:
					save_path=saver.save(sess,'./try.save')
				except:
					pass
			
			print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
