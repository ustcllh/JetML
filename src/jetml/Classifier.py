import torch
import torch.nn as nn
import numpy as np
from .Model import *

class Classifier:
    def __init__(self, model_dict, input_size=3, output_size=2, num_layers=5):
        self.model_dict = model_dict
        self.model = None
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.model = self.load_lstm(self.model_dict)

    def __call__(self, seq):
        structure_t = torch.tensor([seq], dtype=torch.float32)
        out, hidden = self.model(structure_t)
        score = out[-1][-1]
        return nn.functional.softmax(score, dim=0).detach().numpy()

    def load_lstm(self, model_dict):
        model = LSTM(batch_size=1, input_size=self.input_size, output_size=self.output_size, num_layers=self.num_layers)
        model.load_state_dict(torch.load(model_dict, map_location='cpu'))
        model.eval()
        return model
