import torch 
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramLSTM(nn.Module):
  def __init__(self, num_layers, batch_size, num_hidden, num_classes):
    super(SpectrogramLSTM, self).__init__()

    self.num_layers = num_layers
    self.batch_size = batch_size
    self.num_hidden = num_hidden
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(30, 1), stride=1)
    self.bn1 = nn.BatchNorm2d(num_features=32)
    self.pool1 = nn.MaxPool2d(kernel_size=(30, 1), stride=(2, 1))
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(30, 1), stride=1)
    self.bn2 = nn.BatchNorm2d(num_features=32)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(30, 1), stride=1)

    self.lstm1 = nn.LSTM(2240, num_hidden, num_layers, batch_first=True)

    self.linear1 = nn.Linear(num_hidden, num_hidden)
    self.bn3 = nn.BatchNorm1d(num_hidden)
    self.linear2 = nn.Linear(num_hidden, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.conv3(x)

    x = x.view([self.batch_size, -1, x.shape[-1]])
    x = torch.transpose(x, 1, 2)

    out, (h_t, c_t) = self.lstm1(x)
    out = out[:,-1,:]

    out = self.linear1(out)
    out = self.bn3(out)
    out = self.linear2(out)

    return out