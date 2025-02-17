import matplotlib.pyplot as plt
import numpy as np

import scipy.fft

import soundfile as sf
import sounddevice as sd

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

# Using embedded configuration.
separator = Separator('spleeter:2stems')
audio_loader = AudioAdapter.default()
sample_rate = 44100
waveform, _ = audio_loader.load('./data/song1.mp3', sample_rate=sample_rate)

# Perform the separation :
prediction = separator.separate(waveform)
#sf.write('./vocals/song1.wav',prediction,sample_rate)

print("hello")
