import wave as wv
import struct

class wavFile:
	stream = None


	def __init__(self, fileName):
		self.stream = wv.open(fileName, "rb")

	def stream(nSamples):
		num_channels = stream.getnchannels()
		sample_rate = stream.getframerate()
		sample_width = stream.getsampwidth()
		num_frames = stream.getnframes()
		raw_data = stream.readframes(nSamples)

		try:
			if sample_width == 1: 
				fmt = "%iB" % nSamples # read unsigned chars
			elif sample_width == 2:
				fmt = "%ih" % (nSamples) # read signed 2 byte shorts
			else:
				raise ValueError("Only supports 8 and 16 bit audio formats.")
		except:
			return None

		float_data = struct.unpack(fmt, raw_data)
		del raw_data # Keep memory tidy (who knows how big it might be)

		channels = [ [] for time in range(num_channels) ]

		for index, value in enumerate(float_data):
			bucket = index % num_channels
			channels[bucket].append(value)
		first_channel = float_data[::num_channels]
		floatChannel = map(float, first_channel)
	return floatChannel

	def rewind():
		stream.rewind()

	def getLength():
		return stream.getnframes()

def get_wavfile(fileName):
	return waveFile(fileName)
	

def construct_wav(channels, name):
	noise_output = wv.open(name, 'w')
	noise_output.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
	fmt = "%ih" % len(channels)
	data = struct.pack(fmt, *channels)
	noise_output.writeframes(data)
	noise_output.close()



