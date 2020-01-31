import sys
fastjet_dir='/workspace/fastjet/lib/python3.6/site-packages'
sys.path.append(fastjet_dir)
import fastjet as fj
from src.JetML.Event import *
from src.JetTree.JetTree import *

from ROOT import TFile, TTree, TCanvas
from array import array

# case
# 1: pythia
# 2: jewel_NR
# 3: jewel_R
# 4: hybrid
case = 1

# root tree
ofn = 'default.root'

if(case==1):
    ofn = './results/pythia.root'
if(case==2):
    ofn = './results/jewel_NR.root'
if(case==3):
    ofn = './results/jewel_R.root'
if(case==4):
    ofn = './results/hybrid.root'

output = TFile(ofn, 'recreate')
tr = TTree('jet', 'jet substructure' )

nevent_max = 300000

maxn = 100
jetpt = array('f', [0.])
depth = array('i', [0])
zg = array('f', [0.])
deltag = array('f', [0.])
z = array('f', maxn * [0.])
delta = array('f', maxn * [0.])
kperp = array('f', maxn * [0.])


tr.Branch('jetpt', jetpt, 'jetpt/F')
tr.Branch('zg', zg, 'zg/F')
tr.Branch('deltag', deltag, 'deltag/F')
tr.Branch('depth', depth, 'depth/I')
tr.Branch('z', z, 'z[depth]/F')
tr.Branch('delta', delta, 'delta[depth]/F')
tr.Branch('kperp', kperp, 'kperp[depth]/F')


# jet finder
jf = JetFinder(algorithm=fj.antikt_algorithm, R=0.4, ptmin=80)
# jet recluster
jr = JetFinder(algorithm=fj.cambridge_algorithm, R=999., ptmin=0)
# groomer
# sd = SoftDropGroomer(zcut=0.1, beta=0.)
sd = SoftDropGroomer(zcut=0.5, beta=1.5)

if(case==1): # pythia
    input = './pu14/merge/pythia.pu14'
    rd = Reader(input)
    dict, pos = rd.next_event()
    nevent = 0

    while dict['0']:
        if nevent == nevent_max:
            break
        jets = jf(dict['0'])
        for jet in jets:
            constituents = [i for i in jet.constituents()]
            rjets = jr(constituents)
            rjet = rjets[0]
            if abs(rjet.eta())>3:
                continue
            depth[0] = 0
            jtr = JetTree(rjet)
            jetpt[0] = jtr.pseudojet().pt()
            zg[0] = jtr.zg(groomer=sd)
            deltag[0] = jtr.deltag(groomer=sd)
            jtr.groom(groomer=sd, do_recursive_correction=True)
            while jtr.has_structure():
                z[depth[0]] = jtr.z()
                delta[depth[0]] = jtr.delta()
                kperp[depth[0]] = jtr.kperp()
                depth[0] += 1
                jtr = jtr.harder()
            tr.Fill()
        dict, pos = rd.next_event()
        nevent += 1
        if nevent % 100 == 0:
            print('%d events completed!' % nevent)


if(case==2): # jewel NR
    input = './pu14/merge/jewel_NR.pu14'
    rd = Reader(input)
    dict, pos = rd.next_event()
    nevent = 0

    while dict['0']:
        if nevent == nevent_max:
            break
        jets = jf(dict['0'])
        for jet in jets:
            constituents = [i for i in jet.constituents()]
            rjets = jr(constituents)
            rjet = rjets[0]
            if abs(rjet.eta())>3:
                continue
            depth[0] = 0
            jtr = JetTree(rjet)
            jetpt[0] = jtr.pseudojet().pt()
            zg[0] = jtr.zg(groomer=sd)
            deltag[0] = jtr.deltag(groomer=sd)
            jtr.groom(groomer=sd, do_recursive_correction=True)
            while jtr.has_structure():
                z[depth[0]] = jtr.z()
                delta[depth[0]] = jtr.delta()
                kperp[depth[0]] = jtr.kperp()
                depth[0] += 1
                jtr = jtr.harder()
            tr.Fill()
        dict, pos = rd.next_event()
        nevent += 1
        if nevent % 100 == 0:
            print('%d events completed!' % nevent)



if(case==3): # jewel R
    input = './pu14/merge/jewel_R.pu14'
    rd = Reader(input)
    dict, pos = rd.next_event()
    nevent = 0

    while dict['0']:
        if nevent == nevent_max:
            break
        jets = jf(dict['0'])
        ghosts = PseudoJetVec()
        for ghost in dict['-1']:
            ghosts.push_back(ghost)
        gs = GhostSubtractor(ghosts)
        for jet in jets:
            constituents = [i for i in jet.constituents()]
            rjets = jr(constituents)
            rjet = rjets[0]
            if abs(rjet.eta())>3:
                continue
            depth[0] = 0
            jtr = JetTree(rjet, gs=gs)
            jetpt[0] = jtr.pseudojet().pt()
            zg[0] = jtr.zg(groomer=sd)
            deltag[0] = jtr.deltag(groomer=sd)
            jtr.groom(groomer=sd, do_recursive_correction=True)

            while jtr.has_structure():
                z[depth[0]] = jtr.z()
                delta[depth[0]] = jtr.delta()
                kperp[depth[0]] = jtr.kperp()
                depth[0] += 1
                jtr = jtr.harder()
            tr.Fill()
        dict, pos = rd.next_event()
        nevent += 1
        if nevent % 100 == 0:
            print('%d events completed!' % nevent)

if(case==4): # hybrid
    # input = './pu14/hybrid/HYBRID_Hadrons_5020_dijet_K5_kappa_0.404_K_5.000.pu14'
    input = './pu14/hybrid/HYBRID_Hadrons_5020_kappa_0.404_K_0.000.pu14'
    rd = Reader(input)
    dict, pos = rd.next_event()
    nevent = 0

    while dict['0']:
        if nevent == nevent_max:
            break

        # negative wake particles
        ghosts = PseudoJetVec()
        for ghost in dict['2']:
            ghosts.push_back(ghost)
        gs = GhostSubtractor(ghosts, False)

        # particles, positive wake particles, dummies
        dummies = gs.getDummies()
        particles = []
        for p in dict['0']:
            particles.append(p)
        for p in dict['1']:
            particles.append(p)
        for p in dummies:
            particles.append(p)
        jets = jf(particles)

        for jet in jets:
            constituents = [i for i in jet.constituents()]
            rjets = jr(constituents)
            rjet = rjets[0]
            if abs(rjet.eta())>3:
                continue
            depth[0] = 0
            jtr = JetTree(rjet, gs=gs)
            jetpt[0] = jtr.pseudojet().pt()
            zg[0] = jtr.zg(groomer=sd)
            deltag[0] = jtr.deltag(groomer=sd)
            jtr.groom(groomer=sd, do_recursive_correction=True)

            while jtr.has_structure():
                z[depth[0]] = jtr.z()
                delta[depth[0]] = jtr.delta()
                kperp[depth[0]] = jtr.kperp()
                depth[0] += 1
                jtr = jtr.harder()
            tr.Fill()
        dict, pos = rd.next_event()
        nevent += 1
        if nevent % 100 == 0:
            print('%d events completed!' % nevent)

# c = TCanvas('c','z',800, 600)
# tr.Draw('z')
# c.SetLogy()
# c.SaveAs('z.pdf')

output.Write()
output.Close()
