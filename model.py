import torch
import torch.nn as nn


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet,self).__init__()

        self.lstm1 = nn.LSTM(40,100,batch_first = True)

        self.linear1 = nn.Linear(100,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100,10*4)

    def forward(self,x):
        h_0 = torch.zeros((1,16,100)).double().to('cuda')

        c_0 = torch.zeros((1,16,100)).double().to('cuda')
        h_0.requires_grad = True
        c_0.requires_grad = True


        out,(h_t,c_t) = self.lstm1(x,(h_0.detach(),c_0.detach()))

        out = self.linear1(out[:,-1,:])
        out = self.bn1(out)
        out = self.linear2(out)
        out = out.view(-1,10,4)
        return out
