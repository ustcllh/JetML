import sys

# fastjet python module
fastjet_dir='/workspace/fastjet/lib/python3.6/site-packages'
sys.path.append(fastjet_dir)
import fastjet as fj

# constituent subtraction python module
constituent_subtractor_dir='/workspace/ConstituentSubtractor'
sys.path.append(constituent_subtractor_dir)
import ConstituentSubtractor as cs
import IterativeConstituentSubtractor as ics

# jet binary tree python module
from src.JetML.Event import *
from src.JetTree.JetTree import *

# root python module
from ROOT import TFile, TTree, TCanvas
from array import array

# input arguments
# sys.argv[1:7]
#
# CS
# arg1 max_distance
# arg2 alpha
#
# ICS
# arg3 max_distance[0]
# arg4 max_distance[1]
# arg5 alphas[0]
# arg6 alphas[1]
#
# suggested parameters
# 0.25 1 0.2 0.1 1 1
args = sys.argv[0:7]

# print(args)

print('CS Parameters')
print('max_distance: %s' % args[1])
print('alpha: %s' % args[2])

print('CS Parameters')
print('max_distances: [%s,%s]' % (args[3], args[4]))
print('alphas: [%s,%s]' % (args[5], args[6]))

# input file
input = './pu14/pythia_120/PythiaEventsTune14PtHat120_0.pu14'

# thermal background
input_bkg = './pu14/thermal/ThermalEventsMult7000PtAv1.20_0.pu14'

# input_bkg = './pu14/thermal/ThermalEventsMult100PtAv0.85_0.pu14'

print('input file: %s' % input)
print('input background file: %s' % input_bkg)


# root tree
ofn = 'default.root'
output = TFile(ofn, 'recreate')
tr = TTree('jet', 'jet')
tr_cs = TTree('jetcs', 'jet cs')
tr_ics = TTree('jetics', 'jet ics')

nevent_max = 10
maxn = 100

# event variables
x = array('f', [0.])
y = array('f', [0.])
weight = array('f', [0.])
event = array('i', [0])

# jet variables
eta = array('f', [0.])
phi = array('f', [0.])
jetpt = array('f', [0.])
depth = array('i', [0])
zg = array('f', [0.])
deltag = array('f', [0.])
m_groomed = array('f', [0.])
m_ungroomed = array('f', [0.])

# sequential variables
has_structure = array('i', [0])
z = array('f', maxn * [0.])
delta = array('f', maxn * [0.])
kperp = array('f', maxn * [0.])
m = array('f', maxn * [0.])

def set_branches(tr):
    tr.Branch('event', event, 'event/I')
    tr.Branch('x', x, 'x/F')
    tr.Branch('y', y, 'y/F')
    tr.Branch('weight', weight, 'weight/F')

    tr.Branch('eta', eta, 'eta/F')
    tr.Branch('phi', phi, 'phi/F')
    tr.Branch('jetpt', jetpt, 'jetpt/F')
    tr.Branch('depth', depth, 'depth/I')
    tr.Branch('zg', zg, 'zg/F')
    tr.Branch('deltag', deltag, 'deltag/F')
    tr.Branch('m_groomed', m_groomed, 'm_groomed/F')
    tr.Branch('m_ungroomed', m_ungroomed, 'm_ungroomed/F')

    tr.Branch('has_structure', has_structure, 'has_structure/b')
    tr.Branch('z', z, 'z[depth]/F')
    tr.Branch('delta', delta, 'delta[depth]/F')
    tr.Branch('kperp', kperp, 'kperp[depth]/F')
    tr.Branch('m', m, 'm[depth]/F')

set_branches(tr)
set_branches(tr_cs)
set_branches(tr_ics)



# jet finder
jf = JetFinder(algorithm=fj.antikt_algorithm, R=0.4, ptmin=100.)
# jet recluster
jr = JetFinder(algorithm=fj.cambridge_algorithm, R=999., ptmin=0)

# groomer
sd = SoftDropGroomer(zcut=0., beta=0.)
# sd = SoftDropGroomer(zcut=0.1, beta=0.)
# sd = SoftDropGroomer(zcut=0.5, beta=1.5)

# constituent subtractor
# input class: vector<PseudoJet>
# output class: vector<PseudoJet>

def do_clustering(hard_event, ptmin=100.):
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clust_seq= fj.ClusterSequence(hard_event, jet_def)
    jets = clust_seq.inclusive_jets(ptmin)

    return jets, clust_seq

## jet by jet

def do_cs_jet(full_event, ptmin=100.):
    # clustering with ghosts and get the jets
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.7)
    ghost_area = 0.01
    area_def = fj.AreaDefinition(fj.active_area_explicit_ghosts, fj.GhostedAreaSpec(4.0, 1, ghost_area))
    clust_seq_full = fj.ClusterSequenceArea(full_event, jet_def, area_def)
    full_jets = clust_seq_full.inclusive_jets(ptmin)

    # background estimation
    jet_def_for_rho = fj.JetDefinition(fj.kt_algorithm, 0.4)
    rho_range = fj.SelectorAbsRapMax(3.0)
    clust_seq_rho = fj.ClusterSequenceArea(full_event, jet_def, area_def)
    bge_rho = fj.JetMedianBackgroundEstimator(rho_range, jet_def_for_rho, area_def)
    bge_rho.set_jet_density_class(fj.BackgroundJetScalarPtDensity())
    bge_rho.set_particles(full_event)

    # subtractor
    subtractor = cs.ConstituentSubtractor()
    subtractor.set_background_estimator(bge_rho)
    # subtractor.initialize()
    # print(subtractor.description())

    # ghosts = subtractor.get_ghosts()
    # for g in ghosts:
    #     print(g)

    # do subtraction
    # for jet in full_jets:
    #     print(jet)
    #
    # for jet in full_jets:
    #     print(jet)
    #     subtracted_jet = subtractor.result(jet)
    #     print(subtracted_jet)
    #     print("\n")


## event wide

def do_cs_event_wide(full_event, ptmin=100.):
    # cuts
    max_eta = 4.
    max_eta_jet = 3.

    # background estimator
    bge_rho = fj.GridMedianBackgroundEstimator(max_eta, 0.5)
    subtractor = cs.ConstituentSubtractor()
    # enum Distance {
    #     deltaR,
    #     angle
    # }
    subtractor.set_distance_type(0)
    # free parameters: max distance (0.25), alpha (1), ghost area (0.0025)
    subtractor.set_max_distance(float(args[1]))
    subtractor.set_alpha(float(args[2]))
    subtractor.set_ghost_area(0.0025)
    subtractor.set_max_eta(max_eta)
    subtractor.set_background_estimator(bge_rho)
    sel_max_pt = fj.SelectorPtMax(15)
    subtractor.set_particle_selector(sel_max_pt)
    subtractor.initialize()

    # print('Constituent Subtractor')
    # print(subtractor.description())

    bge_rho.set_particles(full_event)

    # clustering
    corrected_event = subtractor.subtract_event(full_event)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clust_seq_corr = fj.ClusterSequence(corrected_event, jet_def)
    corrected_jets = clust_seq_corr.inclusive_jets(ptmin)

    return corrected_jets, clust_seq_corr

## iterative

def do_cs_iterative(full_event, ptmin=100.):
    max_eta = 4.
    max_eta_jet = 3.

    bge_rho = fj.GridMedianBackgroundEstimator(4., 0.5)
    subtractor = ics.IterativeConstituentSubtractor()
    subtractor.set_distance_type(0)

    # max_distances(0.2, 0.1)
    max_distances = ics.DoubleVec()
    max_distances.push_back(float(args[3]))
    max_distances.push_back(float(args[4]))

    # alphas (1, 1)
    alphas = ics.DoubleVec()
    alphas.push_back(float(args[5]))
    alphas.push_back(float(args[6]))

    subtractor.set_parameters(max_distances,alphas)
    subtractor.set_ghost_removal(True)
    subtractor.set_ghost_area(0.0025)
    subtractor.set_max_eta(max_eta);
    subtractor.set_background_estimator(bge_rho)

    sel_max_pt = fj.SelectorPtMax(15)
    # only particles with pt<15 will be corrected - the other particles will be copied without any changes.

    subtractor.set_particle_selector(sel_max_pt)
    subtractor.initialize()

    # print('Iterative Constituent Subtractor')
    # print(subtractor.description())

    bge_rho.set_particles(full_event)

    # clustering
    corrected_event = subtractor.subtract_event(full_event)
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    clust_seq_corr = fj.ClusterSequence(corrected_event, jet_def)
    corrected_jets = clust_seq_corr.inclusive_jets(ptmin)

    return corrected_jets, clust_seq_corr


# processing
rd = Reader(input)
dict, des = rd.next_event()

# thermal event
rd_bkg = Reader(input_bkg)
dict_bkg, des_dkg = rd_bkg.next_event()

# mix event
def mix_event(dict, dict_bkg):
    hard_event = cs.PseudoJetVec()
    full_event = cs.PseudoJetVec()

    hard_event.clear()
    full_event.clear()
    for p in dict['0']:
        hard_event.push_back(p)
        full_event.push_back(p)
    for p in dict_bkg['1']:
        full_event.push_back(p)

    return hard_event, full_event


nevent = 0
while dict['0'] and dict_bkg['1']:
    if nevent == nevent_max:
        break
    event[0] = nevent

    hard_event, full_event = mix_event(dict, dict_bkg)

    # jet pt cut
    ptmin = 130
    ptmin_cs = 100

    # hard event
    jets, clust_seq = do_clustering(hard_event, ptmin)
    for jet in jets:
        constituents = [i for i in jet.constituents()]
        rjets = jr(constituents)
        rjet = rjets[0]
        if abs(rjet.eta())>3:
            continue
        x[0] = 0
        y[0] = 0
        weight[0] = 1
        depth[0] = 0
        jtr = JetTree(rjet)
        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        m_ungroomed[0] = jtr.pseudojet().m()
        zg[0] = jtr.zg(groomer=sd)
        deltag[0] = jtr.deltag(groomer=sd)
        jtr.groom(groomer=sd, do_recursive_correction=False)
        m_groomed[0] = jtr.pseudojet().m()
        if jtr.has_structure():
            has_structure[0] = 1
        else:
            has_structure[0] = 0
        while jtr.has_structure():
            z[depth[0]] = jtr.z()
            delta[depth[0]] = jtr.delta()
            kperp[depth[0]] = jtr.kperp()
            m[depth[0]] = jtr.m()
            depth[0] += 1
            jtr = jtr.harder()
        tr.Fill()

    # jet cs
    corrected_jets, clust_seq_corr = do_cs_event_wide(full_event, ptmin_cs)
    for jet in corrected_jets:
        constituents = [i for i in jet.constituents()]
        rjets = jr(constituents)
        rjet = rjets[0]
        if abs(rjet.eta())>3:
            continue
        x[0] = 0
        y[0] = 0
        weight[0] = 1
        depth[0] = 0
        jtr = JetTree(rjet)
        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        m_ungroomed[0] = jtr.pseudojet().m()
        zg[0] = jtr.zg(groomer=sd)
        deltag[0] = jtr.deltag(groomer=sd)
        jtr.groom(groomer=sd, do_recursive_correction=False)
        m_groomed[0] = jtr.pseudojet().m()
        if jtr.has_structure():
            has_structure[0] = 1
        else:
            has_structure[0] = 0
        while jtr.has_structure():
            z[depth[0]] = jtr.z()
            delta[depth[0]] = jtr.delta()
            kperp[depth[0]] = jtr.kperp()
            m[depth[0]] = jtr.m()
            depth[0] += 1
            jtr = jtr.harder()
        tr_cs.Fill()

    # jet ics
    corrected_jets, clust_seq_corr = do_cs_iterative(full_event, ptmin_cs)
    for jet in corrected_jets:
        constituents = [i for i in jet.constituents()]
        rjets = jr(constituents)
        rjet = rjets[0]
        if abs(rjet.eta())>3:
            continue
        x[0] = 0
        y[0] = 0
        weight[0] = 1
        depth[0] = 0
        jtr = JetTree(rjet)
        eta[0] = jtr.pseudojet().eta()
        phi[0] = jtr.pseudojet().phi()
        jetpt[0] = jtr.pseudojet().pt()
        m_ungroomed[0] = jtr.pseudojet().m()
        zg[0] = jtr.zg(groomer=sd)
        deltag[0] = jtr.deltag(groomer=sd)
        jtr.groom(groomer=sd, do_recursive_correction=False)
        m_groomed[0] = jtr.pseudojet().m()
        if jtr.has_structure():
            has_structure[0] = 1
        else:
            has_structure[0] = 0
        while jtr.has_structure():
            z[depth[0]] = jtr.z()
            delta[depth[0]] = jtr.delta()
            kperp[depth[0]] = jtr.kperp()
            m[depth[0]] = jtr.m()
            depth[0] += 1
            jtr = jtr.harder()
        tr_ics.Fill()

    dict, des = rd.next_event()
    dict_bkg, des_dkg = rd_bkg.next_event()
    nevent += 1
    if nevent % 1000 == 0:
        print('%d events completed!' % nevent)


# c = TCanvas('c','z',800, 600)
# tr.Draw('z')
# c.SetLogy()
# c.SaveAs('z.pdf')

output.Write()
output.Close()
