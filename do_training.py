import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np
import glob


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

class Training_Samples(data.Dataset):
    """
    A customied data loader
    """
    def __init__(self, dir, label):
        self.dir = dir
        self.label = label
        self.data = []
        self.mask = np.array([True, True, False, False, False, True])

        files = glob.glob("%s/*" % self.dir)
        for file in files:
            ifs = open(file, 'r')
            line = ifs.readline()
            while line:
                line = np.fromstring(line, dtype=float, sep=',')
                line = np.reshape(line, (-1, 6))
                self.data.append(line[:, self.mask])
                line = ifs.readline()

        self.len = len(self.data)


    def __getitem__(self, index):
        """
        get a sample from the dataset
        """

        item = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.label, dtype=torch.float32)

        return item, label

    def __len__(self):
        return self.len

def collate_fn_pad(batch):

    seq = [t[0] for t in batch]
    label = [t[1] for t in batch]
    length = [len(i) for i in seq]

    seq = nn.utils.rnn.pad_sequence(seq, batch_first=True)
    label = torch.stack(label)

    return seq, label, length


jewel_training = Training_Samples('./data/cambridge/training/jewel', [1., 0.])

pythia_training = Training_Samples('./data/cambridge/training/pythia', [0., 1.])

print('# of Jets (training): %d jewel jets, %d pythia jets' % (jewel_training.len, pythia_training.len))


data_loader_training = data.DataLoader(
    data.ConcatDataset([
        jewel_training,
        pythia_training
    ]),
    batch_size=100, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
)

# iter = iter(data_loader)
# batch = iter.next()
# print(batch)


# lstm model
class lstm_network(nn.Module):
    # input_size: dim
    # hidden_size: 2 output for classification
    # num_layers: see illustration below
    def __init__(self):
        self.dim = 3
        super(lstm_network, self).__init__()
        self.lstm = nn.LSTM(input_size=self.dim, hidden_size=2, num_layers=5, batch_first=True)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x)
        return x, hidden

    # used for initialization of hidden states and cell states
    def hidden_init(self):
        # (num_layers, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(self.dim, 100, 2))


num_batch = 100

model = lstm_network()
hidden = model.hidden_init()

lossFunction = nn.BCELoss()

# optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# learing rate decay exponentially
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8, last_epoch=-1)

num_epochs = 5


for epoch in range(num_epochs):
    scheduler.step()
    for step, (seq, label, length) in enumerate(data_loader_training):
        out, hidden = model(seq, hidden)

        # cat the output before padding
        res = out[0][length[0]-1]
        for i in range(1, len(length)):
            res = torch.cat((res,out[i][length[i]-1]), dim=0)
        res = res.view(num_batch, -1)
        res = nn.functional.softmax(res, dim=1)

        loss = lossFunction(res,label)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (step+1) % 10 == 0:
            print('Epoch:', epoch,'LR:', scheduler.get_lr())
            print('Eopch: [{}/{}], Step: {}, Loss:{:.2f}'.format(epoch, num_epochs, step+1, loss))

            corr = (res.argmax(dim=1)==label.argmax(dim=1)).sum().item()
            accuracy = 100.00 * corr / num_batch
            print('Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(corr, num_batch, accuracy))



# save model parameters
model_path = './model/lstm_ca_kappa.pt'
torch.save(model.state_dict(), model_path)


# validation

jewel_validation = Training_Samples('./data/cambridge/validation/jewel', [1., 0.])

pythia_validation = Training_Samples('./data/cambridge/validation/pythia', [0., 1.])

print('# of Jets (validation): %d jewel jets, %d pythia jets' % (jewel_validation.len, pythia_validation.len))


data_loader_validation = data.DataLoader(
    data.ConcatDataset([
        jewel_validation,
        pythia_validation
    ]),
    batch_size=100, shuffle=True, num_workers=4, drop_last=True, collate_fn=collate_fn_pad
)


corr = 0
total = 0

for step, (seq, label, length) in enumerate(data_loader_validation):

    out, hidden = model(seq, hidden)

    res = out[0][length[0]-1]
    for i in range(1, len(length)):
        res = torch.cat((res,out[i][length[i]-1]), dim=0)
    res = res.view(num_batch, -1)
    res = nn.functional.softmax(res, dim=1)

    corr += (res.argmax(dim=1)==label.argmax(dim=1)).sum().item()
    total += num_batch


accuracy = 100.00 * corr / total
print('Validation Accuracy[Corr/Total]: [{}/{}] = {:.2f} %' .format(corr, total, accuracy))
