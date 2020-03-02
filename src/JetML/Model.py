import torch
import torch.nn as nn
import numpy as np


# lstm model
class LSTM(nn.Module):
    # input_size: dim
    # hidden_size: 2 output for classification
    # num_layers: see illustration below
    def __init__(self, input_size=3, output_size=2, num_layers=5):
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.output_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, x):
        batch_size = len(x)
        h_0 = self.hidden_init(batch_size)
        c_0 = self.hidden_init(batch_size)
        x, hidden = self.lstm(x, (h_0, c_0))
        return x, hidden

    # used for initialization of hidden states and cell states
    def hidden_init(self, batch_size):
        # h_0, c_0
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.output_size))
