import torch
import torch.nn as nn


class CustomNet(nn.Module):

    def __init__(self, device='cuda', n_layers = 1, h_size = 100):
        super(CustomNet,self).__init__()

        self.lstm1 = nn.LSTM(input_size=40,hidden_size=h_size,num_layers=n_layers,batch_first = True)
        self.linear1 = nn.Linear(h_size,3)
        self.lstm2 = nn.LSTM(input_size=40,hidden_size=h_size,num_layers=n_layers,batch_first = True)
        self.linear2 = nn.Linear(h_size,3)
        self.lstm3 = nn.LSTM(input_size=40,hidden_size=h_size,num_layers=n_layers,batch_first = True)
        self.linear3 = nn.Linear(h_size,3)
        self.lstm4 = nn.LSTM(input_size=40,hidden_size=h_size,num_layers=n_layers,batch_first = True)
        self.linear4 = nn.Linear(h_size,3)

        self.nlayers = n_layers
        self.hsize = h_size
        self.device = device

    def forward(self,x):

        h_01 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        c_01 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        h_01.requires_grad = True
        c_01.requires_grad = True

        h_02 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        c_02 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        h_02.requires_grad = True
        c_02.requires_grad = True

        h_03 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        c_03 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        h_03.requires_grad = True
        c_03.requires_grad = True

        h_04 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        c_04 = torch.zeros((self.nlayers,x.size(0),self.hsize)).double().to(self.device)
        h_04.requires_grad = True
        c_04.requires_grad = True

        out1,(h_t,c_t) = self.lstm1(x,(h_01.detach(),c_01.detach()))
        out2,(h_t,c_t) = self.lstm2(x,(h_02.detach(),c_02.detach()))
        out3,(h_t,c_t) = self.lstm3(x,(h_03.detach(),c_03.detach()))
        out4,(h_t,c_t) = self.lstm4(x,(h_04.detach(),c_04.detach()))

        out1 = self.linear1(out1[:,-1,:])
        out2 = self.linear2(out2[:,-1,:])
        out3 = self.linear3(out3[:,-1,:])
        out4 = self.linear4(out4[:,-1,:])

        out1 = torch.unsqueeze(out1,2)
        out2 = torch.unsqueeze(out2,2)
        out3 = torch.unsqueeze(out3,2)
        out4 = torch.unsqueeze(out4,2)
        out = torch.cat((out1,out2,out3,out4), 2)
        #out = out.view(-1,3,4)
        return out
