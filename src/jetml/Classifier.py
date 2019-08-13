import torch
import torch.nn as nn
from .Reader import *
from .Transformer import *


class Jet_Classifier:
    def __init__(self, model):
        self.model_name = model
        self.model = None
        self.transformer = None
        if self.model_name is 'cnn':
            self.model = self._load_cnn_()
            self.transformer = Jet_Transformer(model='cnn')
        if self.model_name is 'lstm':
            self.model = self._load_lstm_()
            self.transformer = Jet_Transformer(model='lstm')

    def __call__(self, jet):
        if not self.model:
            print('Model None')
        if self.model_name is 'cnn':
            img = self.transformer(jet)
            img_t = torch.tensor(img, dtype=torch.float32)
            return self.model(img_t)
        if self.model_name is 'lstm':
            seq = self.transformer(jet)
            seq_t = torch.tensor(seq, dtype=torch.float32)
            hidden = torch.autograd.Variable(torch.zeros(5, 1, 2))
            return self.model(seq_t, hidden)

    def _load_cnn_(self):
        cnn_conf = './model/cnn.pt'
        model = cnn(90, 130, 150, 200, 1)
        model.load_state_dict(torch.load(cnn_conf, map_location='cpu'))
        model.eval()
        return model

    def _load_lstm_(self):
        lstm_conf = './model/lstm.pt'
        model = lstm()
        model.load_state_dict(torch.load(lstm_conf, map_location='cpu'))
        model.eval()
        return model


class cnn(nn.Module):
    def __init__(self, a, b, c, d, batch_size):
        super(cnn, self).__init__()
        # 33x33 -> 15x15
        self.conv1 = nn.Conv2d(5, a, kernel_size=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1_drop = nn.Dropout2d(p=0.1)

        # 15x15 -> 6x6
        self.conv2 = nn.Conv2d(a, b, kernel_size=4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_drop = nn.Dropout2d(p=0.1)

        # 6x6 -> 2x2
        self.conv3 = nn.Conv2d(b, c, kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv3_drop = nn.Dropout2d(p=0.1)

        self.fc1 = nn.Linear(c*2*2, d)
        self.fc2 = nn.Linear(d, 2)
        self.relu = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.conv1_drop(x)
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = self.conv2_drop(x)
        x = self.relu(self.maxpool3(self.conv3(x)))
        x = self.conv3_drop(x)

        x = x.view(self.batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x).detach().numpy()

class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=2, num_layers=5, batch_first=True)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x)
        score = x[-1][-1]
        return nn.functional.softmax(score, dim=0).detach().numpy()

    def hidden_init(self):
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(5, 100, 2))
