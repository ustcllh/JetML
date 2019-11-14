from src.jetml.Event import *
from src.jetml.Jet import *

import os, sys
fastjet_path = '/workspace/fastjet/lib/python3.7/site-packages/fastjet.py'
sys.path.append(os.path.dirname(fastjet_path))

import fastjet as fj

import sys

if len(sys.argv)!=3:
    print("Usage: python prepare_training_data.py <input pu14> <output filename>")
    sys.exit()

ifn = sys.argv[1]
ofn = sys.argv[2]

# reader = Reader('/workspace/samples/PythiaEventsTune14PtHat120_0.pu14')
reader = Reader(ifn)
ofs= open(ofn,"w+")

kt = []

event = reader.next_event()
nevent = 0

while event:
    jf = Jet_Finder()
    jets, partons = jf(event)
    for i in jets:
        # algorithms:   fj.antikt_algorithm fj.cambridge_algorithm
        jet = Jet(i, algorithm=fj.cambridge_algorithm, R=1000.).pseudojet
        jet_tr = JetTree(jet)
        structure = jet_tr.primary_structure()

        if not structure:
            continue

        for i in structure:
            for j in i:
                if j==i[0]:
                    ofs.write('%f,' % j)
                elif j==i[-1]:
                    ofs.write('%f' % j)
                    if not i==structure[-1]:
                        ofs.write(',')
                else:
                    ofs.write('%f,' % j)
        ofs.write('\n')
    nevent += 1
    event = reader.next_event()
    if nevent%100==0:
        print('%d events finisshed' % nevent)
ofs.close()
