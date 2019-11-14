import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import glob

class Training_Samples(data.Dataset):
    """
    A customied data loader
    """
    def __init__(self, dir, label, mask):
        self.dir = dir
        self.label = label
        self.data = []
        # self.mask = np.array([True, True, False, False, False, True])
        self.mask = mask

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
