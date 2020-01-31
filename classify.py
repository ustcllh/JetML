import torch
import torch.nn as nn
import numpy as np
from array import array
from src.JetML.Classifier import *
from ROOT import TFile, TTree


ifn = "./results/ptmin80/jewel_R_zcut0p5_beta1p5.root"
ifs = TFile(ifn, 'READ')
itr = ifs.Get("jet")

prefix = 'test'

lstm = array('f', [0.])
ofn = './results/' + prefix + '.root'
ofs = TFile(ofn, 'recreate')

otr = TTree('jet', 'lstm' )
otr.Branch('lstm', lstm, 'lstm/F')


jc = Classifier('./model/hybrid_zcut0p5_beta1p5.pt')


for entry in itr:
    lstm[0] = -999
    depth = entry.depth
    if depth!=0:
        z = entry.z
        delta = entry.delta
        kperp = entry.kperp
        item = []
        for i in range(depth):
            item.append([z[i], delta[i], kperp[i]])
        lstm[0] = jc(item)[0]
    otr.Fill()

ofs.Write()
ofs.Close()
