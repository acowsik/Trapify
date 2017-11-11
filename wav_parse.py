import wave as wv
import struct


def parse_wav(wave_file):
	stream = wv.open(wave_file,"rb")

	num_channels = stream.getnchannels()
	sample_rate = stream.getframerate()
	sample_width = stream.getsampwidth()
	num_frames = stream.getnframes()

	raw_data = stream.readframes( num_frames ) # Returns byte data
	stream.close()

	total_samples = num_frames * num_channels

	if sample_width == 1: 
		fmt = "%iB" % total_samples # read unsigned chars
	elif sample_width == 2:
		fmt = "%ih" % (total_samples) # read signed 2 byte shorts
	else:
		raise ValueError("Only supports 8 and 16 bit audio formats.")

	float_data = struct.unpack(fmt, raw_data)
	del raw_data # Keep memory tidy (who knows how big it might be)

	channels = [ [] for time in range(num_channels) ]

	for index, value in enumerate(float_data):
		bucket = index % num_channels
		channels[bucket].append(value)
	floatChannel = []
	for i in channels[0]:
		floatChannel.append(float(i))
	print floatChannel
	return floatChannel

def construct_wav(channels, name):
	noise_output = wv.open(name, 'w')
	noise_output.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
	fmt = "%if" % len(channels)
	data = struct.pack(fmt, *channels)
	noise_output.writeframes(data)
	noise_output.close()



