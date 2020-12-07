 class SpectrogramLSTM(nn.Module):
  def __init__(self, num_layers, num_hidden, num_classes):
    super(SpectrogramLSTM, self).__init__()

    self.num_layers = num_layers
    self.num_hidden = num_hidden
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 1), stride=1, padding=(1, 0))
    self.bn1 = nn.BatchNorm2d(num_features=32)
    self.pool1 = nn.MaxPool2d(kernel_size=(30, 1), stride=(2, 1))
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(30, 1), stride=1)
    self.bn2 = nn.BatchNorm2d(num_features=32)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=1)

    self.lstm1 = nn.LSTM(160, num_hidden, num_layers, batch_first=True)

    self.linear1 = nn.Linear(num_hidden, num_hidden)
    self.bn3 = nn.BatchNorm1d(num_hidden)
    self.linear2 = nn.Linear(num_hidden, 3*num_classes)

  def forward(self, x):
    # print(x.shape)
    # x = self.conv1(x)
    # x = self.bn1(x)
    # x = self.pool1(x)
    # x = F.relu(self.conv2(x))
    # x = self.bn2(x)
    # x = F.relu(self.conv3(x))
    # print(x.shape)

    x = x.view([x.shape[0], -1, x.shape[-1]])
    x = torch.transpose(x, 1, 2)
    # print(x.shape)
    out, (h_t, c_t) = self.lstm1(x)
    out = out[:,-1,:]

    out = self.linear1(out)
    out = self.bn3(out)
    out = self.linear2(out)
    out = out.view(-1,3,3)
    return out
