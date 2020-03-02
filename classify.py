import torch
import torch.nn as nn
import numpy as np
from array import array
from src.JetML.Classifier import *
from ROOT import TFile, TTree


ifn = "./results/ptmin130/pythia.root"
# ifn = "./results/hybrid.root"

ifs = TFile(ifn, 'READ')
itr = ifs.Get("jet")

# prefix = 'jewel_NR_false_zcut0p5_beta1p5'
prefix = 'pythia_lstm'


ofn = './results/' + prefix + '.root'
ofs = TFile(ofn, 'recreate')

maxn = 100
x = array('f', [0.])
y = array('f', [0.])
eta = array('f', [0.])
phi = array('f', [0.])
weight = array('f', [0.])
jetpt = array('f', [0.])
depth = array('i', [0])
zg = array('f', [0.])
deltag = array('f', [0.])
z = array('f', maxn * [0.])
delta = array('f', maxn * [0.])
kperp = array('f', maxn * [0.])
m = array('f', maxn * [0.])
lstm_hybrid = array('f', [0.])
lstm_jewel_R = array('f', [0.])
lstm_jewel_NR = array('f', [0.])
m_groomed = array('f', [0.])
m_ungroomed = array('f', [0.])
has_structure = array('b', [0])

otr = TTree('jet', 'lstm' )
otr.Branch('lstm_hybrid', lstm_hybrid, 'lstm_hybrid/F')
otr.Branch('lstm_jewel_R', lstm_jewel_R, 'lstm_jewel_R/F')
otr.Branch('lstm_jewel_NR', lstm_jewel_NR, 'lstm_jewel_NR/F')
otr.Branch('x', x, 'x/F')
otr.Branch('y', y, 'y/F')
otr.Branch('eta', eta, 'eta/F')
otr.Branch('phi', phi, 'phi/F')
otr.Branch('weight', weight, 'weight/F')
otr.Branch('jetpt', jetpt, 'jetpt/F')
otr.Branch('zg', zg, 'zg/F')
otr.Branch('deltag', deltag, 'deltag/F')
otr.Branch('depth', depth, 'depth/I')
otr.Branch('m_groomed', m_groomed, 'm_groomed/F')
otr.Branch('m_ungroomed', m_ungroomed, 'm_ungroomed/F')
otr.Branch('z', z, 'z[depth]/F')
otr.Branch('delta', delta, 'delta[depth]/F')
otr.Branch('kperp', kperp, 'kperp[depth]/F')
otr.Branch('has_structure', has_structure, 'has_structure/b')
otr.Branch('m', m, 'm[depth]/F')

jc_hybrid = Classifier('./model/hybrid.pt')
jc_jewel_R = Classifier('./model/jewel_R.pt')
jc_jewel_NR = Classifier('./model/jewel_NR.pt')

for entry in itr:
    lstm_hybrid[0] = -999
    lstm_jewel_R[0] = -999
    lstm_jewel_NR[0] = -999

    # x[0] = entry.x
    # y[0] = entry.y
    # eta[0] = entry.eta
    # phi[0] = entry.phi
    weight[0] = entry.weight
    jetpt[0] = entry.jetpt
    zg[0] = entry.zg
    deltag[0] = entry.deltag
    depth[0] = entry.depth
    m_groomed[0] = entry.m_groomed
    m_ungroomed[0] = entry.m_ungroomed
    for i in range(entry.depth):
        z[i] = entry.z[i]
        delta[i] = entry.delta[i]
        kperp[i] = entry.kperp[i]
        m[i] = entry.m[i]

    if entry.depth!=0:
        item = []
        for i in range(entry.depth):
            item.append([z[i], delta[i], kperp[i]])
        lstm_hybrid[0] = jc_hybrid(item)[0]
        lstm_jewel_R[0] = jc_jewel_R(item)[0]
        lstm_jewel_NR[0] = jc_jewel_NR(item)[0]
    otr.Fill()

ofs.Write()
ofs.Close()
