import torch
import torch.nn as nn
import numpy as np


# lstm model
class lstm_network(nn.Module):
    # input_size: dim
    # hidden_size: 2 output for classification
    # num_layers: see illustration below
    def __init__(self, dim):
        self.dim = dim

        super(lstm_network, self).__init__()
        self.lstm = nn.LSTM(input_size=self.dim, hidden_size=2, num_layers=5, batch_first=True)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x)
        return x, hidden

    # used for initialization of hidden states and cell states
    def hidden_init(self):
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(5, 100, 2))
