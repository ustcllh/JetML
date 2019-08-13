import os, sys
fastjet_path = '/workspace/JetML/src/jetml'
sys.path.append(os.path.dirname(fastjet_path))

from jetml import *

ifn = '/workspace/samples/jewel_1.pu14'
ofn = './data/jewel_btr.txt'

ofs = open(ofn,"w")

rd = Reader(ifn)

event=rd.next_event()
njet = 0
nevent = 0
while event:
    jf = Jet_Finder()
    jets = jf(event)
    for jet in jets:
        jet_tr = JetTree(jet)
        while jet_tr and jet_tr.lundCoord:
            state = jet_tr.state()
            ofs.write('%d\t' % njet)
            for var in state:
                ofs.write('%f\t' % var)
            ofs.write('\n')
            jet_tr = jet_tr.harder
        njet += 1

    if (nevent % 100 == 0):
        print('nEvent: %d' % nevent)
    event = rd.next_event()
    nevent += 1
