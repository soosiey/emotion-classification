from scipy.signal import stft
import numpy as np
from tqdm import tqdm
import utils
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = utils.load_eeg('dataset/')
    d = data['data']
    l = data['labels']
    print(d.shape, l.shape)

    d_stft = np.zeros((1280, 32, 313, 35))
    fs = 134.4
    f = None
    for video in range(1280):
      for channel in range(32):
        # Get the magnitude spectrogram
        f, t, Zxx = stft(d[video][channel], fs=fs, nperseg=1024, noverlap=784)
        # A bandpass frequency filter from 4.0-45.0Hz was applied.
        lowest_freq, highest_freq = np.where(f >= 4)[0][0] - 1, np.where(f <= 45)[0][-1] + 1
        Zxx = abs(Zxx[lowest_freq:highest_freq])
        d_stft[video][channel] = Zxx

    d_train, d_test, l_train, l_test = train_test_split(d_stft, l, test_size=0.20, random_state=42)

    np.save('d_train_stft.npy', d_train)
    np.save('d_test_stft.npy', d_test)
    np.save('l_train_stft.npy', l_train)
    np.save('l_test_stft.npy', l_test)

    bins = np.array([1, 1+8/3, 1+16/3, 9.1])
    l_train_3bins = np.digitize(l_train, bins) - 1
    l_test_3bins = np.digitize(l_test, bins) - 1
    np.save('l_train_3bins.npy', l_train_3bins)
    np.save('l_test_3bins.npy', l_test_3bins)

    np.save('freq.npy', f)
