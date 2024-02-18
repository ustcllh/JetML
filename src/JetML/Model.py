import torch
import torch.nn as nn
import numpy as np

# lstm-fc
class LSTM_FC(nn.Module):
    def __init__(self, input_size=4, hidden_size=[20,4], num_layers=2, batch_size=1, device=torch.device('cpu')):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        h0 = self.hidden_init().to(self.device)
        c0 = self.hidden_init().to(self.device)
        self.hidden = (h0, c0)

        print('************** Model ****************')
        print('Neural Network:\tLSTM')
        print('Input Size:\t%d' % self.input_size) 
        print('hidden Size:\t[%d,%d]' % (self.hidden_size[0], self.hidden_size[1])) 
        print('No. of Layers:\t%d' % self.num_layers) 
        print('Batch Size:\t%d' % self.batch_size) 
        print('*************************************')

        super(LSTM_FC, self).__init__()

        # lstm + 2FC
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], num_layers=self.num_layers, batch_first=True, dropout=0.05).to(self.device)
        self.fc1 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1], bias=True).to(self.device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.hidden_size[1], out_features=2, bias=True).to(self.device)


    def forward(self, x):
        # lstm + 2FC
        x, hidden = self.lstm(x, self.hidden)
        x = [i[-1] for i in x]
        x = self.fc1(torch.stack(x, dim=0).to(self.device))
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # used for initialization of hidden states and cell states
    def hidden_init(self):
        # h_0, c_0
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size[0]))

