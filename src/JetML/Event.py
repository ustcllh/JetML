import sys
fastjet_dir='/workspace/fastjet/lib/python3.6/site-packages'
sys.path.append(fastjet_dir)


import fastjet as fj
from math import pow, sqrt
from math import cosh, tanh, sin, cos


class Reader:
    """
    Reader for pu14 data samples.
    """
    def __init__(self, input):
        self.stream = open(input)
        self.stream.readline()
        self.count = 0

    def __next__(self):
        particles=[]
        description=[]
        dict = {}
        while 1:
            line = self.stream.readline()
            if not line:
                break
            if('end' in line):
                self.count += 1
                break
            if 'weight' in line:
                description = line.split()
                continue
            if not 'event' in line:
                particles.append(line.split())

        dict['-1'] = self.sift(particles, -1)
        dict['0'] = self.sift(particles, 0)
        dict['1'] = self.sift(particles, 1)
        dict['2'] = self.sift(particles, 2)
        des = self.get_description(description)

        return dict, des

    def next_event(self):
        return self.__next__()

    def sift(self, particles, status=0):
        res = []
        for particle in particles:
            if (int(particle[5])==status):
                res.append(self.pseudojet(particle))
        return res

    @staticmethod
    def get_description(description):
        des = {}
        try:
            id_x = description.index('X')
            x = description[id_x + 1]
            des['X'] = x
        except:
            des['X'] = 0

        try:
            id_y = description.index('Y')
            y = description[id_y + 1]
            des['Y'] = y
        except:
            des['Y'] = 0

        try:
            id_weight = description.index('weight')
            des['weight'] = description[id_weight + 1]
        except:
            des['weight'] = 1

        return des

    @staticmethod
    def pseudojet(particle):
        p = [float(item) for item in particle]
        e = pow(p[0], 2) + pow(p[1], 2) + pow(p[2], 2) + pow(p[3], 2)
        e = sqrt(e)
        temp = fj.PseudoJet(p[0], p[1], p[2], e)
        temp.set_user_index(int(p[4]))
        return temp


class JetFinder:
    """
    do jet clustering using antikt algorithm (default)
    fj.cambridge_algorithm
    fj.antikt_algorithm
    """
    def __init__(self, algorithm=fj.antikt_algorithm, R=0.4, ptmin=120):
        self.algorithm = algorithm
        self.R = R
        self.ptmin = ptmin
        self.particles = []
        self.dummies = []
        self.jets = []

    def __call__(self, particles, dummies=[]):
        self.jets = []
        self.particles = particles
        self.dummies = dummies
        self.particles = self.particles + self.dummies
        self.do_clustering()

        return self.jets

    def do_clustering(self):
        self.jet_def = fj.JetDefinition(self.algorithm, self.R)
        self.jet_cs = fj.ClusterSequence(self.particles, self.jet_def)
        jets = self.jet_cs.inclusive_jets(self.ptmin)
        self.jets = fj.sorted_by_pt(jets)

        return self.jets

def getDummies(ghosts):
    res = []
    for ghost in ghosts:

        eta = ghost.eta()
        phi = ghost.phi()
        e = 1e-6
        px = e * cos(phi) / cosh(eta)
        py = e * sin(phi) / cosh(eta)
        pz = e * tanh(eta)
        res.append(fj.PseudoJet(px, py, pz, e))
        return res
