import torch
import numpy as np
from torch.utils.data import Dataset


class EmotionDataset(Dataset):


    def __init__(self, train=True, transform=None):

        self.data = np.load('xtrain.npy') if train else np.load('xtest.npy')
        self.labels = np.load('ytrain.npy') if train else np.load('ytest.npy')
        tmp = np.zeros(self.labels.shape)
        tmp[self.labels < 3] = 0
        tmp[(self.labels >= 3) & (self.labels < 7)] = 1
        tmp[self.labels >= 7] = 2
        self.labels = tmp.astype(int)
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
