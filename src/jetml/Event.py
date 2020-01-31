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
        pos = self.get_position(description)

        return dict, pos

    def next_event(self):
        return self.__next__()

    def sift(self, particles, status=0):
        res = []
        for particle in particles:
            if (int(particle[5])==status):
                res.append(self.pseudojet(particle))
        return res

    @staticmethod
    def get_position(description):
        try:
            id_x = description.index('X')
            id_y = description.index('Y')
            x = description[id_x + 1]
            y = description[id_y + 1]
            return [x, y]
        except:
            return []

    @staticmethod
    def pseudojet(particle):
        p = [float(item) for item in particle]
        e = pow(p[0], 2) + pow(p[1], 2) + pow(p[2], 2) + pow(p[3], 2)
        e = sqrt(e)
        temp = fj.PseudoJet(p[0], p[1], p[2], e)
        temp.set_user_index(int(p[3]))
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

        # try:
        eta = ghost.eta()
        phi = ghost.phi()
        e = 1e-6
        px = e * cos(phi) / cosh(eta)
        py = e * sin(phi) / cosh(eta)
        pz = e * tanh(eta)
        res.append(fj.PseudoJet(px, py, pz, e))
        return res
        # except:
        #     eta = ghost.eta()
        #     phi = ghost.phi()
        #     print("Dummy Error!")
        #     print("eta, phi = %f, %f" % (eta, phi))
        #     exit
        #     return res


if __name__ == '__main__':
    # rd = Reader('/workspace/JetML/pu14/hybrid/HYBRID_Hadrons_5020_dijet_K5_kappa_0.404_K_5.000.pu14')

    rd = Reader('/workspace/JetML/pu14/jewel_parton/jewel_1.pu14')
    particles_dict, pos = rd.next_event()
    print(particles_dict['-1'])
