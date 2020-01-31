import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from ROOT import TFile

class Training_Samples(data.Dataset):
    """
    A customied data loader
    """
    def __init__(self, file, label, events=[0, 20000]):
        self.file = file
        self.label = label
        self.data = []

        f = TFile(self.file, "READ")
        tr = f.Get("jet")

        low = events[0]
        up = events[1]
        max = tr.GetEntriesFast()

        idx = low
        while (idx<up and idx<max):
            tr.GetEntry(idx)
            depth = tr.depth
            if depth==0:
                idx += 1
                continue
            z = tr.z
            delta = tr.delta
            kperp = tr.kperp
            item = []
            for i in range(depth):
                item.append([z[i], delta[i], kperp[i]])
            self.data.append(item)
            idx += 1


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
