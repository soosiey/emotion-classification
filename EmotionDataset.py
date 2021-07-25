import torch
import numpy as np
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, train=True, transform=None):

        self.data = np.load('d_train_stft.npy') if train else np.load('d_test_stft.npy')
        self.labels = np.load('l_train_3bins.npy') if train else np.load('l_test_3bins.npy')

        self.data = self.data.astype(np.double)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        sample = self.data[idx,:]
        labels = self.labels[idx,:]
        if(self.transform):
            sample = self.transform(sample)

        return {'data':sample,'labels':labels}
