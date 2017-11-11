import numpy as np


CHUNK_SIZE = 2048

def split_phase_and_power(transform):
    """
    Takes the raw fourier transform and returns a tuple of (power, phase, scale_factor)
    Power is the absolute value of transform / scale_factor such that sum(power) = 1
    phase is the phase of each value in the transform
    """
    power = np.abs(transform)
    phase = np.angle(transform)
    scale_factor = np.max(power)
    if scale_factor == 0:
        return power, phase, 1
    else:
        return power / scale_factor, phase, scale_factor

def combine_phase_and_power(power, phase, scale_factor):
    """
    Combines the phase and power and scale factor components
    """
    waveform = np.multiply(power, np.exp(1j * phase))
    return np.multiply(power, np.exp(1j * phase)) * scale_factor
    
def chunkify(song):
    """
    An iterable that returns the chunks of the fourier transform
    """
    assert len(song) >= CHUNK_SIZE * 2
    for i in xrange(0, len(song) - CHUNK_SIZE, CHUNK_SIZE // 2):
        yield np.fft.rfft(song[i: i + CHUNK_SIZE])

def unchunkify(chunks):
    """
    Takes the cunks and recombines them
    """
    recreated_chunks = list(map(lambda x: np.fft.irfft(combine_phase_and_power(*x)), chunks))
    total_length = len(recreated_chunks) * CHUNK_SIZE // 2
    output = np.zeros(total_length)
    window = np.power(np.sin(np.linspace(0, np.pi, CHUNK_SIZE)), 2)
    
    for i, j in enumerate(xrange(0, total_length - CHUNK_SIZE, CHUNK_SIZE // 2)):
        o = window * recreated_chunks[i]
            
        output[j: j+CHUNK_SIZE] += o
    return output

def trapify(power):
    return power

def process(song):
    chunks = chunkify(song)
    split_chunks = map(split_phase_and_power, chunks)
    trapified = map(lambda c: (trapify(c[0]), c[1], c[2]), split_chunks)
    new_song = unchunkify(trapified)
    return new_song