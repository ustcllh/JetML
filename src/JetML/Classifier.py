import torch
import torch.nn as nn
import numpy as np
import json
from .Model import *

class Classifier:
    def __init__(self, model_name, input_size=4):
        self.model_name = model_name
        self.input_size = input_size
        with open(self.model_name + '.json', 'r') as f:
            self.dict = json.load(f)
            self.hidden_size = [int(self.dict['hidden_size0']), int(self.dict['hidden_size1'])]
            self.num_layers = int(self.dict['num_layers'])

        self.model = self.load_lstm(self.model_name + '.pt')

    def __call__(self, seq):
        structure_t = torch.tensor([seq], dtype=torch.float32)
        out = self.model(structure_t)
        score = out[-1]
        return nn.functional.softmax(score, dim=0).detach().numpy()

    def load_lstm(self, model_dict):
        model = LSTM_FC(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        model.load_state_dict(torch.load(model_dict, map_location='cpu'))
        model.eval()
        return model
