import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
import crepe
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
        if file == 'Auto_Tuned_Vocal_is.wav' or file == 'Auto_Tuned_Vocal.wav' or file == 'Auto_Tuned.wav':
            training_autotuned.append(path)
        elif file == 'Original_Vocal_is.wav' or file == 'Original_Vocal.wav' or file == 'Original.wav':
            training_original.append(path)
for root, dirs, files in os.walk('vocal_data/test'):
    for file in files:
        path = os.path.join(root, file)
        if file == 'Auto_Tuned_Vocal_is.wav' or file == 'Auto_Tuned_Vocal.wav' or file == 'Auto_Tuned.wav':
            test_autotuned.append(path)
        elif file == 'Original_Vocal_is.wav' or file == 'Original_Vocal.wav' or file == 'Original.wav':
            test_original.append(path)

data = [('image_data/training/autotuned/autotuned', training_autotuned), ('image_data/training/original/original', training_original),
        ('image_data/test/autotuned/autotuned', test_autotuned), ('image_data/test/original/original', test_original)]

for path, dataset in data:
    for i in range(len(dataset)):
        file = dataset[i]
        Y, fs = librosa.load(file)

        # Dataset Generation
        seg_len = 10 * fs
        segs = [Y[j:j + seg_len] for j in range(0, len(Y), seg_len)]
        #f0_segs = [frequency[i:i + len(frequency) // len(segs)] for i in range(0, len(frequency), len(frequency) // len(segs))]

        # Convert segments to mel-spectrograms
        mel_spectrograms = []
        for k in range(len(segs)):
            s = segs[k]
            print(path, file, i, k)
            time, frequency, confidence, activation = crepe.predict(s, fs, viterbi=True)
            mel_spec = librosa.feature.melspectrogram(y=s, sr=fs, n_fft=2048, hop_length=1024, n_mels=128)
            mel_spectrograms.append(mel_spec)

            # Plot and save
            plt.figure(figsize=(4, 4))
            librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=fs, hop_length=1024, y_axis="mel", x_axis="time")

            # Save figure
            plt.axis('off')
            plt.plot(time, frequency, label='Pitch (Hz)', color='w')
            plt.savefig(f"{path}_{i}_{k}.png", dpi=300, bbox_inches='tight', pad_inches = 0)
            plt.close()
            