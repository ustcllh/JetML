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
            if not line:
                break
            if('end' in line):
                self.count += 1
                break
            if 'event' in line or 'weight' in line:
                continue
            event.append(line.split())
        return event

    def next_event(self):
        return self.__next__()

class Jet_Finder:
    """
    do jet clustering
    """
    def __init__(self, algorithm=fj.antikt_algorithm, R=0.4, ptmin=120):
        self.algorithm = algorithm
        self.R = R
        self.ptmin = ptmin
        self.jets = []

    def __call__(self, event):
        self.__set_event__(event)
        self.__do_clustering()
        return self.get_jets(), self.get_partons()

    def __set_event__(self, event):
        self.event = event
        self.jets = []

    def __do_clustering(self):
        particles = []
        partons = []
        for particle in self.event:
            p = [float(item) for item in particle]
            e = pow(p[0], 2) + pow(p[1], 2) + pow(p[2], 2) + pow(p[3], 2)
            e = sqrt(e)
            if p[5]==0:
                particles.append(fj.PseudoJet(p[0], p[1], p[2], e))
            elif p[5]==1:
                parton = fj.PseudoJet(p[0], p[1], p[2], e)
                if p[4]==21:
                    parton.set_user_index(21)
                else:
                    parton.set_user_index(0)
                partons.append(parton)
        self.jet_def = fj.JetDefinition(self.algorithm, self.R)
        self.jet_cs = fj.ClusterSequence(particles, self.jet_def)
        jets = self.jet_cs.inclusive_jets(self.ptmin)
        self.jets = fj.sorted_by_pt(jets)
        self.partons = partons

    def get_jets(self):
        return self.jets

    def get_partons(self):
        return self.partons


if __name__ == '__main__':
    rd = Reader('/workspace/samples/PythiaEventsTune14PtHat120_0.pu14')
    event = rd.next_event()
    jf = Jet_Finder()

    jets = jf(event)
    for jet in jets:
        print(jet.pt())
