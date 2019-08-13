from src.jetml import *

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import math



datafile = '/workspace/samples/jewel_dijet_NR_pthat120_v221p.pu14'
"""
datafile = '/workspace/samples/pythia_dijet_pthat120_private.pu14'
"""


ofs = open("./res/results.txt","w")
ofs_btr = open("./res/btree.txt","w")

ofs_groomed = open("./res/results_groomed.txt","w")
ofs_btr_groomed = open("./res/btree_groomed.txt","w")

rd = Reader(datafile)

def analyze(jet):
    jc1 = Jet_Classifier('lstm')
    jc2 = Jet_Classifier('cnn')

    # ML
    lstm = jc1(jet)[0]
    cnn = jc2(jet)[0][0]

    # jet
    jet_tr = JetTree(jet)

    vec = jet_tr.node
    px = vec[0]
    py = vec[1]
    pz = vec[2]
    e = vec[3]
    eta = jet.eta()
    phi = jet.phi()
    jet_m = math.exp(jet_tr.lundCoord.lnm)
    jet_z = math.exp(jet_tr.lundCoord.lnz)
    jet_delta = math.exp(jet_tr.lundCoord.lnDelta)
    jet_angularity = jet_tr.angularity.values()
    res = [px, py, pz, e, eta, phi, lstm, cnn, jet_m, jet_z, jet_delta, jet_angularity[0], jet_angularity[1], jet_angularity[2]]

    # jet binary tree
    jet_btr = jet_tr.binary_tree(append=[lstm, cnn])

    # groomed tree
    # groomer = RSD(zcut=0.1, beta=0., R0=0.4)
    # groomer = RSD(zcut=0.5, beta=1.5, R0=0.4)
    groomer = RSD(zcut=0.5, beta=1.5, R0=0.4)


    jet_tr_groomed = groomer(jet_tr)
    vec_groomed = jet_tr_groomed.node
    px_groomed = vec_groomed[0]
    py_groomed = vec_groomed[1]
    pz_groomed = vec_groomed[2]
    e_groomed = vec_groomed[3]
    if jet_tr_groomed.lundCoord:
        jet_m_groomed = math.exp(jet_tr_groomed.lundCoord.lnm)
        jet_z_groomed = math.exp(jet_tr_groomed.lundCoord.lnz)
        jet_delta_groomed = math.exp(jet_tr_groomed.lundCoord.lnDelta)
        res_groomed = [jet_m_groomed, jet_z_groomed, jet_delta_groomed]
    else:
        res_groomed = [-9999, -9999, -9999]
    res_groomed = [px_groomed, py_groomed, pz_groomed, e_groomed] + [lstm, cnn] + res_groomed

    jet_btr_groomed = jet_tr_groomed.binary_tree(append=[lstm, cnn])

    return res, jet_btr, res_groomed, jet_btr_groomed

event=rd.next_event()
count = 0
while event:
    jf = Jet_Finder(ptmin=130)
    jets, partons = jf(event)
    for jet in jets:
        if abs(jet.eta())>3:
            continue
        ratio = -9999
        dr = -9999
        pid = -9999
        for parton in partons:
            dr_temp = jet.delta_R(parton)
            if dr_temp<0.4:
                ratio = jet.pt()/parton.pt()
                dr = dr_temp
                pid = parton.user_index()
        res, jet_btr, res_groomed, jet_btr_groomed = analyze(jet)
        res.append(ratio)
        res.append(dr)
        res.append(pid)

        for val in res:
            ofs.write('%f\t' % val)
        ofs.write('\n')

        for vals in jet_btr:
            if vals[0][1] == 0:
                continue
            for val in vals[1]:
                ofs_btr.write('%f\t' % val)
            ofs_btr.write('\n')

        for val in res_groomed:
            ofs_groomed.write('%f\t' % val)
        ofs_groomed.write('\n')

        for vals in jet_btr_groomed:
            if vals[0][1] == 0:
                continue
            for val in vals[1]:
                ofs_btr_groomed.write('%f\t' % val)
            ofs_btr_groomed.write('\n')



    event=rd.next_event()
    count+=1
    if(count%100==0):
        print(count)
    # if(count==200):
    #     break
