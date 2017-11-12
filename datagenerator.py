import numpy as np
from wav_parse import wavFile
from trapify import split_phase_and_power

"""
forms a data generator that can produce the next batch of
testing or training images along with their labels
"""

class DataGenerator:
	def __init__(self, filepath, outSize, inSize):
		self.INSIZE = inSize
		self.OUTSIZE = outSize
		self.wave = wavFile(filepath)
		
	def next_batch(self, batch_size):
		# function returns the next batch of images as a np array of 4d size
		# returns the labels in a numpy array as well
		
		r = self.wave.stream(batch_size*self.INSIZE)

		if r == None:
			self.wave.rewind()
			r = self.wave.stream(batch_size*self.INSIZE)

		rNum = np.reshape(r, [batch_size, self.INSIZE])

		converted = []

		for i in rNum:
			converted.append(split_phase_and_power(i)[0])

		return np.array(converted)
		
	def length(self):
		return self.wave.getLength()

				
"""
# testing
gen = ImageDataGenerator('training.txt', 2506)
a = gen.next_batch(50)
print "img shape is %s, label shape is %s" % (a[0].shape, a[1].shape)

img shape is (50, 227, 227, 3), label shape is (50, 2506)
"""

