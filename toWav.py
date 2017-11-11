"""
dependencies:
pydub
ffmpeg
"""

from pydub import AudioSegment

def mp3toWav(fileName):
	song = AudioSegment.from_mp3(fileName)
	song.export(fileName.split(".")[0]+".wav", format="wav")