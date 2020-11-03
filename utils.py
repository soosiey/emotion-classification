import pickle
import numpy as np


def load_eeg():
    return pickle.load(open('dataset/data_preprocessed_python/s01.dat','rb'), encoding='latin1')

if __name__ == '__main__':
    data = load_eeg()
    print('Data length:', len(data))
    print('Data keys:', data.keys())
    for key in data.keys():
        print(key,'shape:',data[key].shape)
