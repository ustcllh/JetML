import torch
import torch.nn as nn
import numpy as np
from .Jet import *
from .Model import *

class Jet_Classifier:
    def __init__(self, model_dict, mask):
        self.model_dict = model_dict
        self.model = None
        self.mask = mask
        self.dim = np.sum(mask)
        self.model = self.load_lstm(model_dict)

    def __call__(self, pseudojet):
        if not self.model_dict:
            print('Model Not Defined')
            exit
        else:
            jet = Jet(pseudojet)
            jet_tr = JetTree(jet.pseudojet)
            structure = np.array(jet_tr.primary_structure())
            if len(structure)==0:
                return []
            structure = np.array([structure[:, self.mask]])
            seq_t = torch.tensor(structure, dtype=torch.float32)
            # (num_layers, batch_size, hidden_size)
            hidden = torch.autograd.Variable(torch.zeros(5, 1, 2))
            out, hidden = self.model(seq_t, hidden)
            score = out[-1][-1]
            return nn.functional.softmax(score, dim=0).detach().numpy()

    def load_lstm(self, model_dict):
        model = lstm_network(self.dim)
        model.load_state_dict(torch.load(model_dict, map_location='cpu'))
        model.eval()
        return model
