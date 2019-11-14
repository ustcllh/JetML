from src.jetml.Event import *
from src.jetml.Jet import *
from src.jetml.Classifier import *
import os, sys
fastjet_path = '/workspace/fastjet/lib/python3.7/site-packages/fastjet.py'
sys.path.append(os.path.dirname(fastjet_path))

import fastjet as fj
import numpy as np

if len(sys.argv)!=3:
    print("Usage: python do_analysis.py <input pu14> <output filename prefix>")
    sys.exit()

ifn = sys.argv[1]
ofn_prefix = sys.argv[2]

ofn_jet = ofn_prefix + '_jet'
ofn_strc = ofn_prefix + '_strc'

reader = Reader(ifn)
ofs_jet= open(ofn_jet,"w+")
ofs_strc= open(ofn_strc,"w+")

event = reader.next_event()
nevent = 0

mask = np.array([True, True, True, True, False, False])
jc = Jet_Classifier('./model/lstm_11110.pt', mask)

while event:
    jf = Jet_Finder()
    jets, partons = jf(event)

    for i in jets:
        # algorithms:   fj.antikt_algorithm fj.cambridge_algorithm
        jet = Jet(i, algorithm=fj.cambridge_algorithm, R=1000.).pseudojet
        jet_tr = JetTree(jet)
        ps = jet_tr.primary_structure()
        fs = jet_tr.full_structure()
        if not ps:
            continue
        lstm = jc(i)
        for j in ps:
            for k in j:
                if k==j[0]:
                    ofs_strc.write('%f,' % k)
                elif k==j[-1]:
                    ofs_strc.write('%f' % k)
                    if not j==ps[-1]:
                        ofs_strc.write(',')
                else:
                    ofs_strc.write('%f,' % k)

        ratio = -9999
        dr = -9999
        pid = -9999
        for parton in partons:
            dr_temp = i.delta_R(parton)
            if dr_temp<0.4:
                ratio = i.pt()/parton.pt()
                dr = dr_temp
                pid = parton.user_index()


        px = jet.px()
        py = jet.py()
        pz = jet.pz()
        e = jet.e()
        eta = jet.eta()
        phi = jet.phi()
        ang = cal_angularity(i)

        ofs_jet.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%f,%f,%f' % (px, py, pz, e, eta, phi, lstm[0], ratio, dr, pid, ang[0], ang[1], ang[2]))
        ofs_strc.write('\n')
        ofs_jet.write('\n')
    nevent += 1
    event = reader.next_event()
    if nevent%100==0:
        print('%d events finished' % nevent)

ofs_jet.close()
ofs_strc.close()
