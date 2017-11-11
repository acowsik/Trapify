import tensorflow as tf
import numpy as np
"""
Alexnet Architecture 
"""
class AutoEncoder(object):
	
	def __init__(self, in_size, out_size, hid_layers):
		# instance variables
		self.IN_SIZE = in_size
		self.OUT_SIZE = out_size
		self.HID_1 = hid_layers[0]
		self.HID_2 = hid_layers[1]
		self.HID_3 = hid_layers[2]
		
		# weights
		self.enc_h1 = tf.Variable(tf.random_normal([in_size, hid_1]), name = "enc_h1")
		self.enc_h2 = tf.Variable(tf.random_normal([hid_1, hid_2]), name = "enc_h2")
		self.enc_h3 = tf.Variable(tf.random_normal([hid_2, hid_3]), name = "enc_h3")

		self.dec_h1 = tf.Variable(tf.random_normal([hid_3, hid_2]), name = "dec_h1")
		self.dec_h2 = tf.Variable(tf.random_normal([hid_2, hid_1]), name = "dec_h2")
		self.dec_h1 = tf.Variable(tf.random_normal([hid_1, out_size]), name = "dec_h3")

		#biases
		self.enc_b1 = tf.Variable(tf.random_normal([hid_1]), name = "enc_b1")
		self.enc_b2 = tf.Variable(tf.random_normal([hid_2]), name = "enc_b2")
		self.enc_b3 = tf.Variable(tf.random_normal([hid_3]), name = "enc_b3")

		self.dec_b1 = tf.Variable(tf.random_normal([hid_2]), name = "dec_b1")
		self.dec_b2 = tf.Variable(tf.random_normal([hid_1]), name = "dec_b2")
		self.dec_b3 = tf.Variable(tf.random_normal([out_size]), name = "dec_b3")
			
	def run(self, x, keep_prob):
		with tf.device("/gpu:0"):

			"""
			Encode:
			1st layer: FC, RELU in_size -> hid_1
			2nd layer: FC, RELU hid_1 -> hid_2
			3rd layer: FC, RELU hid_2 -> hid_3

			Decode:
			1st layer: FC, RELU hid_3 -> hid_2
			2nd layer: FC, RELU hid_2 -> hid_1
			3rd layer: FC, Sigmoid? hid_1->out_size

			CHECK IF DROPOUT NEEDED
			"""

			encode_1 = fc(self.X, self.enc_h1, self.enc_b1 self.IN_SIZE, self.HID_1, relu = True)
			encode_2 = fc(encode_1, self.enc_h2, self.enc_b2, self.HID_1, self.HID_2, relu = True)
			middle = fc(encode_2, self.enc_h3, self.enc_b3, self.HID_2, self.HID_3, relu = True)

			decode_1 = fc(middle, self.dec_h1, self.dec_b1, self.HID_3, self.HID_2, relu = True)
			decode_2 = fc(decode_1, self.dec_h2, self.dec_b2, self.HID_2, self.HID_1, relu = True)
			end = fc(decode_2, self.dec_h3, self.dec_b3, self.HID_1, self.OUT_SIZE, relu = False)
		
		return end
	
	
	"""
	Predefining all layers
	"""
def fc(x, w, b, num_in, num_out, relu = False):
	
	# Matrix multiplication
	act = tf.nn.xw_plus_b(x, w, b)
	
	if relu:
		# applying ReLu non linearity
		relu = tf.nn.relu(act)
		return relu
	else:
		return act

def lrn(x, radius, alpha, beta, name, bias = 1.0):
	return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
						beta = beta, bias = bias, name = name)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

