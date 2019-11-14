import os, sys
fastjet_path = '/workspace/fastjet/lib/python3.7/site-packages/fastjet.py'
sys.path.append(os.path.dirname(fastjet_path))

import fastjet as fj
import math

def do_clustering(particles, algorithm=fj.cambridge_algorithm, R=1000.):
    """
    jet clustering
    """
    jet_def = fj.JetDefinition(algorithm, R)
    return jet_def(particles)

def do_groom(pseudojet, zcut=0.5, beta=1., R0=0.4):
    """
    soft drop groom
    """
    particles = ()
    if not pseudojet:
        return particles
    j1 = fj.PseudoJet()
    j2 = fj.PseudoJet()
    if pseudojet.has_parents(j1,j2):
        # order the parents in pt
        if (j2.pt() > j1.pt()):
            j1,j2=j2,j1

        # soft drop condition
        z = j2.pt()/(j1.pt() + j2.pt())
        delta = j1.delta_R(j2)
        soft_drop = (z < zcut * math.pow(delta/R0, beta))

        particles = particles + do_groom(j1, zcut, beta, R0)
        if not soft_drop:
            particles = particles + do_groom(j2, zcut, beta, R0)
    else:
        return pseudojet.constituents()
    return particles

def cal_angularity(pseudojet):
    ptd = 0
    mass = 0
    width = 0
    R = 0.4
    phi = pseudojet.phi()
    eta = pseudojet.eta()
    pt = pseudojet.pt()
    for particle in pseudojet.constituents():
        dr = math.sqrt(math.pow(particle.eta()-eta, 2) + math.pow(particle.phi()-phi, 2))
        ptd += math.pow(particle.pt()/pt, 2)
        width += particle.pt()/pt * dr/R
        mass += particle.pt()/pt * math.pow(dr/R, 2)
    return [ptd, mass, width]

class Jet:
    """
    Jet:
    1. pseudojet
    2. groomed pseudojet
    """
    def __init__(self, pseudojet, algorithm=fj.cambridge_algorithm, R=1000.):
        self.pseudojet = do_clustering(pseudojet.constituents(), algorithm, R)[0]

        """
        pseudojet groomed with soft drop
        """
        particles_groomed = do_groom(self.pseudojet)
        self.pseudojet_groomed = do_clustering(particles_groomed, algorithm, R)[0]

    def pseudojet(self):
        return self.pseudojet

    def pseudojet_groomed(self):
        return self.pseudojet_groomed

    def angularity(self):
        return cal_angularity(self.pseudojet)

    def angularity_groomed(self):
        return cal_angularity(self.pseudojet_groomed)


class JetTree:
    """
    Jet binary tree representation
    """
    def __init__(self, pseudojet, child=None):
        """Initialize a new node, and create its two parents if they exist."""
        self.harder = None
        self.softer = None
        self.child  = child
        self.state  = []
        j1 = fj.PseudoJet()
        j2 = fj.PseudoJet()
        if pseudojet and pseudojet.has_parents(j1,j2):
            # order the parents in pt
            if (j2.pt() > j1.pt()):
                j1,j2=j2,j1

            # branching variables
            delta = j1.delta_R(j2)
            z     = j2.pt()/(j1.pt() + j2.pt())
            lnm     = 0.5*math.log(abs((j1 + j2).m2()))
            lnKt    = math.log(j2.pt()*delta)
            lnz     = math.log(z)
            lnDelta = math.log(delta)
            lnKappa = math.log(z*delta)
            psi     = math.atan((j1.rap() - j2.rap())/(j1.phi() - j2.phi()))
            self.state = [lnz, lnDelta, lnKt, lnm, psi, lnKappa]
            # self.state = [lnz, lnDelta, lnKappa, lnm, psi]
            # then create two new tree nodes with j1 and j2
            self.harder = JetTree(j1, self)
            self.softer = JetTree(j2, self)

    def primary_structure(self):
        ps = []
        if self.state:
            ps.append(self.state)
            ps += self.harder.primary_structure()
        return ps

    def full_structure(self):
        fs = []
        if self.state:
            fs.append(self.state)
            fs += self.harder.full_structure()
            fs += self.softer.full_structure()
        return fs
