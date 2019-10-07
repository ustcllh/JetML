import torch
import torch.nn as nn
import numpy as np
from .Jet import *


class Jet_Classifier:
    def __init__(self, model, mask):
        self.model_name = model
        self.model = None
        self.transformer = None
        self.mask = mask
        # cnn model depreciated
        if self.model_name is 'lstm':
            self.model = self.load_lstm()

    def __call__(self, pseudojet):
        if self.model_name is 'lstm':

            jet = Jet(pseudojet)
            jet_tr = JetTree(jet.pseudojet)
            structure = np.array(jet_tr.primary_structure())
            structure = np.array([structure[:, self.mask]])

            seq_t = torch.tensor(structure, dtype=torch.float32)
            hidden = torch.autograd.Variable(torch.zeros(1, 1, 2))
            return self.model(seq_t, hidden)
        else:
            print('Model Not Defined')

    def load_lstm(self):
        lstm_conf = './model/lstm_ca_00100.pt'
        model = lstm()
        model.load_state_dict(torch.load(lstm_conf, map_location='cpu'))
        model.eval()
        return model

class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=2, num_layers=5, batch_first=True)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x)
        score = x[-1][-1]
        return nn.functional.softmax(score, dim=0).detach().numpy()

    def hidden_init(self):
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(1, 100, 2))
