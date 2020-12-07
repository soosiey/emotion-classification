import numpy as np


def bin_power(X, Band, Fs):
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))): 
                int(np.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio

def load_eeg(d):
    data = []
    labels = []
    for f in tqdm(range(1,33)):
        if(f < 10):
            s = '0'+str(f)
        else:
            s = str(f)
        tmp = pickle.load(open(d+'data_preprocessed_python/'+'s'+s+'.dat','rb'),encoding='latin1')
        data.append(tmp['data'])
        labels.append(tmp['labels'])
    ret = {'data':np.concatenate(data,axis=0),'labels':np.concatenate(labels,axis=0)}

    return ret

def download_stft(d, l):
    # #trials x #channels x #frequencies x #timesteps
    d_stft = np.zeros((1280, 32, 5, 488))
    l_stft = np.zeros((1280, 4))
    for participant in range(32):
      for video in range(40):
        video_idx = participant * 40 + video
        data = d[video_idx]
        labels = l[video_idx]
        
        l_stft[video_idx] = labels
        start = 0
        t = 0
        while start + window_size < data.shape[1]:
          for channel in range(32):
            X = data[channel][start : start + window_size]
            Power, Power_Ratio = bin_power(X, band, sample_rate)
            d_stft[video_idx, channel,:,t] = Power_Ratio
      
          start = start + step_size
          t += 1

        print(video_idx, "done")

    d_train, d_test, l_train, l_test = train_test_split(d_stft, l_stft, test_size=0.20, random_state=42)
    np.save('d_train_stft.npy', d_train)
    np.save('d_test_stft.npy', d_test)
    np.save('l_train_stft.npy', l_train)
    np.save('l_test_stft.npy', l_test)


if __name__ == '__main__':
    channel = np.arange(32)
    band = [4,8,12,16,25,45] #5 bands
    window_size = 256 #Averaging band power of 2 sec
    step_size = 16 #Each 0.125 sec update once
    sample_rate = 128 #Sampling rate of 128 Hz

    data = load_eeg('')
    d = data['data']
    l = data['labels']
    print(d.shape, l.shape)

    download_stft(d, l)