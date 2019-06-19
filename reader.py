import os, sys
fastjet_path = '/workspace/fastjet/lib/python3.7/site-packages/fastjet.py'
sys.path.append(os.path.dirname(fastjet_path))

import fastjet as fj
from math import pow, sqrt

class Reader:
    """
    Reader for pu14 data samples.
    """
    def __init__(self, input, nev=-1):
        self.nev = nev
        self.stream = open(input)
        self.stream.readline()
        self.count = 0

    def __next__(self):
        event=[]
        while 1:
            line = self.stream.readline()
            if('end' in line):
                self.count += 1
                break
            if 'weight' in line:
                continue
            if not 'event' in line:
                event.append(line.split())

        return event

    def next_event(self):
        return self.__next__()

class Jet_Finder:
    """
    do jet clustering
    """
    def __init__(self, algorithm=fj.antikt_algorithm, R=0.4):
        self.algorithm = algorithm
        self.R = R
        self.jets = []

    def __call__(self, event):
        self.__set_event__(event)
        return self.get_jets()

    def __set_event__(self, event):
        self.event = event
        self.jets = []

    def __do_clustering(self):
        particles = []
        for particle in self.event:
            p = [float(item) for item in particle]
            e = pow(p[0], 2) + pow(p[1], 2) + pow(p[2], 2) + pow(p[3], 2)
            e = sqrt(e)
            particles.append(fj.PseudoJet(p[0], p[1], p[2], e))
        jet_def = fj.JetDefinition(self.algorithm, self.R)
        jets = jet_def(particles)
        self.jets = fj.sorted_by_pt(jets)

    def get_jets(self):
        self.__do_clustering()
        return self.jets

if __name__ == '__main__':
    rd = Reader('/workspace/samples/PythiaEventsTune14PtHat120_0.pu14')
    event = rd.next_event()
    jf = Jet_Finder()

    jets = jf(event)
    for jet in jets:
        print(jet.pt())
