import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import crepe
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from PIL import Image
from tempfile import TemporaryDirectory
# Generating All Datasets
# Auto_Tuned_Vocal_is.wav
# Auto_Tuned_Vocal.wav
# Auto_Tuned.wav

# Original_Vocal_is.wav
# Original_Vocal.wav
# Original.wav

training_autotuned = []
training_original = []
test_autotuned = []
test_original = []

# Iterate through directory
for root, dirs, files in os.walk('vocal_data/training'):
    for file in files:
        path = os.path.join(root, file)
        if file == 'Auto_Tuned.wav':
            training_autotuned.append(path)
        elif file == 'Original.wav':
            training_original.append(path)
for root, dirs, files in os.walk('vocal_data/test'):
    for file in files:
        path = os.path.join(root, file)
        if file == 'Auto_Tuned.wav':
            test_autotuned.append(path)
        elif file == 'Original.wav':
            test_original.append(path)

data = [('image_data2/training/autotuned/autotuned', training_autotuned), ('image_data2/training/original/original', training_original),
        ('image_data2/test/autotuned/autotuned', test_autotuned), ('image_data2/test/original/original', test_original)]

file = 'Viva_La_Vida_Coldplay/Coldplay - Viva La Vida.mp3'
Y, fs = librosa.load(file)

# Dataset Generation
seg_len = 10 * fs
segs = [Y[j:j + seg_len] for j in range(0, len(Y), seg_len)]
#f0_segs = [frequency[i:i + len(frequency) // len(segs)] for i in range(0, len(frequency), len(frequency) // len(segs))]

# Convert segments to mel-spectrograms
mel_spectrograms = []
for k in range(len(segs)):
    s = segs[k]
    #print(path, file, i, k)
    time, frequency, confidence, activation = crepe.predict(s, fs, viterbi=True)
    mel_spec = librosa.feature.melspectrogram(y=s, sr=fs, n_fft=2048, hop_length=1024, n_mels=128)
    mel_spectrograms.append(mel_spec)

    # Plot and save
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=fs, hop_length=1024, y_axis="mel", x_axis="time")

    # Save figure
    plt.axis('off')
    plt.plot(time, frequency, label='Pitch (Hz)', color='w')
    plt.savefig(f"Viva_La_Vida_Coldplay/plots/plot_{k}.png", dpi=300, bbox_inches='tight', pad_inches = 0)
    plt.close()
