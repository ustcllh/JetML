import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from ROOT import TFile
from math import log

class Samples(data.Dataset):
    """
    A customied data loader
    """
    def __init__(self, file, tree, label, events=[0, 20000], zg_cut=0.1, rg_cut=0.1, jetpt_cut=200, weight_factor=1e7):
        self.file = file
        self.label = label
        self.data = []
        self.weight = []
        self.weight_sum = 0
        self.zg_cut = zg_cut
        self.rg_cut = rg_cut
        self.jetpt_cut = jetpt_cut
        self.weight_factor = weight_factor

        f = TFile(self.file, "READ")
        tr = f.Get(tree)

        low = events[0]
        up = events[1]
        max = tr.GetEntriesFast()

        idx = low
        while (idx<up and idx<max):
            tr.GetEntry(idx)
            depth = tr.depth
            if depth==0 or tr.jetpt<self.jetpt_cut or tr.delta[0]<self.rg_cut or tr.z[0]<self.zg_cut:
                idx += 1
                continue
            z = tr.z
            delta = tr.delta
            kperp = tr.kperp
            m = tr.m
            self.weight.append(tr.weight)
            self.weight_sum += tr.weight
            item = []
            for i in range(depth):
                item.append([z[i], delta[i], kperp[i], m[i]])
            self.data.append(item)
            idx += 1


        self.len = len(self.data)


    def __getitem__(self, index):
        """
        get a sample from the dataset
        """

        item = torch.tensor(self.data[index], dtype=torch.float32)
        weight = torch.tensor([self.weight[index]*self.weight_factor/self.weight_sum], dtype=torch.float32)
        label = torch.tensor(self.label, dtype=torch.float32)

        return item, weight, label

    def __len__(self):
        return self.len


def collate_fn_pad(batch):
    seq = [t[0] for t in batch]
    weight = [t[1] for t in batch]
    label = [t[2] for t in batch]
    length = [len(i) for i in seq]

    seq = nn.utils.rnn.pad_sequence(seq, batch_first=True)
    weight = torch.stack(weight)
    label = torch.stack(label)
    return seq, weight, label, length
